from   collections import namedtuple, Counter, defaultdict
from   dataclasses import dataclass, field
import logging
import os
import pickle
import random
from   PIL import Image, ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from   typing import Optional, List, Callable, Union, Dict, Any
import warnings
from   time import perf_counter
import re
import hashlib
import multiprocessing.dummy as mp
from copy import deepcopy

import numpy as np

from .masks import make_mask_from_data, DEFAULT_MASK_EXTRA_RADIUS
from .splits import taskonomy_flat_split_to_buildings
from .transforms import default_loader, get_transform, convertrgb2lab
from .task_configs import task_parameters, SINGLE_IMAGE_TASKS #, *

ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this

LabelFile = namedtuple('LabelFile', ['point', 'view', 'domain'])
View = namedtuple('View', ['building', 'point', 'view'])

RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)


class TaskonomyDataset(data.Dataset):
    '''
        Loads data for the Taskonomy dataset.
        This expects that the data is structured
        
            /path/to/data/
                rgb/
                    modelk/
                        point_i_view_j.png
                        ...                        
                depth_euclidean/
                ... (other tasks)
                
        If one would like to use pretrained representations, then they can be added into the directory as:
            /path/to/data/
                rgb_encoding/
                    modelk/
                        point_i_view_j.npy
                ...
        
        Basically, any other folder name will work as long as it is named the same way.
    '''
    @dataclass
    class Options():
        '''
            data_path: Path to data
            tasks: Which tasks to load. Any subfolder will work as long as data is named accordingly
            buildings: Which models to include. See `splits.taskonomy` (can also be a string, e.g. 'fullplus-val')
            transform: one transform per task.
            
            Note: This assumes that all images are present in all (used) subfolders
        '''
        data_path: str = '/PATH_TO/taskonomy'
        tasks: List[str] = field(default_factory=lambda: ['rgb'])
        buildings: Union[List[str], str] = 'tiny'
        transform: Optional[Union[Dict[str, Callable], str]] = "DEFAULT"  # Dict[str, Transform], None, "DEFAULT"
        do_center_crop_transform: bool = False  # Dict[str, Transform], None, "DEFAULT"
        image_size: Optional[int] = None
        normalize_rgb: bool = False
        force_refresh_tmp: bool = False
        file_name_parser: Optional[Union[Dict[str, Callable], str]] = "DEFAULT" # NO_MULTIVIEW
        max_images: Optional[int] = None
        shared_cache: Dict[str, Any] = None
        data_seed: Optional[int] = -1
        rgb2lab: Optional[bool] = False

    def __init__(self, options: Options):
        start_time = perf_counter()

        if isinstance(options.tasks, str):
            options.tasks = [options.tasks]
            options.transform = {options.tasks: options.transform}            
        
        self.buildings = taskonomy_flat_split_to_buildings[options.buildings] if isinstance(options.buildings, str) else options.buildings
        self.data_path = options.data_path
        self.image_size = options.image_size
        self.tasks = options.tasks
        self.normalize_rgb = options.normalize_rgb
        self.force_refresh_tmp = options.force_refresh_tmp
        self.file_name_parser = options.file_name_parser
        if self.file_name_parser == "DEFAULT":
            self.file_name_parser = default_file_name_parser
        elif self.file_name_parser == "NO_MULTIVIEW":
            self.file_name_parser = no_multiview_file_name_parser
        self.max_images = options.max_images
        self.loader_threadpool = None
        self.shared_cache = options.shared_cache
        self.data_seed = options.data_seed
        self.rgb2lab = options.rgb2lab

        # Load saved image locations if they exist, otherwise create and save them
        tmp_path = get_cached_fpaths_filename(
            [t if t != '2d_edges' else 'depth_zbuffer' for t in options.tasks],
            options.buildings
        )
        # tmp_path = get_cached_fpaths_filename('depth_zbuffer-mask_valid-rgb'.split('-'), options.buildings)
        tmp_exists = os.path.exists(tmp_path)
        if tmp_exists and not self.force_refresh_tmp:
            with open(tmp_path, 'rb') as f:
                self.urls = pickle.load(f)
            self.size = len(self.urls[self.tasks[-1]])
            print(f'Loaded TaskonomyDataset with {self.size} images from tmp.')
        else:
            self.urls = {task: make_dataset(os.path.join(self.data_path, task), self.buildings)
                        for task in options.tasks if task != '2d_edges'}
            self.urls, self.size  = self._remove_unmatched_images()

            # Save extracted URLs
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, 'wb') as f:
                pickle.dump(self.urls, f)

        self.transform = options.transform
        if isinstance(self.transform, str):
            if self.transform == 'DEFAULT':
                self.transform = {task: get_transform(task, self.image_size) for task in self.tasks}
            else:
                raise ValueError('TaskonomyDataset option transform must be a Dict[str, Callable], None, or "DEFAULT"')

        if options.do_center_crop_transform:
            self.transform = {
                k: transforms.Compose(
                    t.transforms + [transforms.CenterCrop(self.image_size)] 
                )
                for k, t in self.transform.items()
            }

        if self.normalize_rgb and 'rgb' in self.transform:
            self.transform['rgb'] = transforms.Compose(
                self.transform['rgb'].transforms +
                [transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)]
            )

        
        # Saving some lists and dictionaries for fast lookup

        self.tbpv_dict = {} # Save task -> building -> point -> view dict
        self.url_dict = {}  # Save (task, building, point, view) -> URL dict
        self.bpv_count = {} # Dictionary to check if all (building, point, view) tuples have all tasks
        bpv_tasks = [t for t in self.tasks if t != '2d_edges']
        for task in bpv_tasks:
            self.tbpv_dict[task] = {}
            for url in self.urls[task]:
                building, point, view = self.file_name_parser(url)
                # building = url.split('/')[-2]
                # file_name = url.split('/')[-1].split('_')
                # point, view = file_name[1], file_name[3]

                # Populate url_dict
                self.url_dict[(task, building, point, view)] = url

                # Populate tbpv_dict
                if building not in self.tbpv_dict[task]:
                    self.tbpv_dict[task][building] = {}
                if point not in self.tbpv_dict[task][building]:
                    self.tbpv_dict[task][building][point] = []
                self.tbpv_dict[task][building][point].append(view)

                # Populate bpv_count
                if (building, point, view) not in self.bpv_count:
                    self.bpv_count[(building, point, view)] = 1
                else:
                    self.bpv_count[(building, point, view)] += 1

        # Remove entries that don't have all tasks and create list of all (building, point, view) tuples that contain all tasks
        self.bpv_list = [bpv_tuple for bpv_tuple, count in self.bpv_count.items() if count == len(bpv_tasks)]

        self.views = {}    # Build dictionary that contains all the views from a certain (building, point) tuple
        self.bpv_dict = {} # Save building -> point -> view dict
        for building, point, view in self.bpv_list:
            # Populate views
            if (building, point) not in self.views:
                self.views[(building, point)] = []
            self.views[(building, point)].append(view)

            # Populate bpv_dict
            if building not in self.bpv_dict:
                self.bpv_dict[building] = {}
            if point not in self.bpv_dict[building]:
                self.bpv_dict[building][point] = []
            self.bpv_dict[building][point].append(view)

        end_time = perf_counter()
        self.num_points = len(self.views)
        self.num_images = len(self.bpv_list)
        self.num_buildings = len(self.bpv_dict)
        
        logger = logging.getLogger(__name__)
        logger.warning("Loaded {} images from {} in {:0.2f} seconds".format(self.num_images, self.data_path, end_time - start_time))
        logger.warning("\t ({} buildings) ({} points) ({} images) for domains {}".format(self.num_buildings, self.num_points, self.num_images, self.tasks))

        # Shuffle dataset in a repeatable manner
        self.randomize_order()

        # Limit number of images if requested
        if self.max_images:
            self.bpv_list = self.bpv_list[:self.max_images]
            print(f'Using {len(self.bpv_list)} images')


    def __len__(self):
        return len(self.bpv_list)

    # @torch.jit.script
    def __getitem__(self, index):

        #  Building / point / view
        building, point, view = self.bpv_list[index]
        # fn_args = [(task, building, point, view) for task in self.tasks]

        # if self.loader_threadpool is None:
        #     self.loader_threadpool = mp.Pool(len(self.tasks))
        # results = self.loader_threadpool.starmap(self._getitem_single_domain, fn_args)

        # result = {task: res for task, res in zip(self.tasks, results)}

        result = {}
        for task in self.tasks:
            num_channels = task_parameters[task]['out_channels']

            if task == '2d_edges':
                path = self.url_dict[('rgb', building, point, view)]
            else:
                path = self.url_dict[(task, building, point, view)]

            if self.shared_cache is not None:
                if path in self.shared_cache:
                    res = self.shared_cache[path]
                else:
                    res = default_loader(path)
                    self.shared_cache[path] = deepcopy(res)
            else:
                res = default_loader(path)

            # Convert RGB to Lab
            if 'rgb' in task and self.rgb2lab:
                assert res.mode == 'RGB'

                # resize
                if self.image_size is not None:
                    transform_ = transforms.Resize(self.image_size)
                    res = transform_(res)

                # convert rgb to lab
                res = convertrgb2lab(res)
    
                # For this L channel image, do not use default transform
                # use the following transform to transform it into [-50, 50]
                res = np.array(res)
                res = res.astype(np.float32)
                # convert to 0-100 first, and then subtract 50.0
                # TODO: check if normalize to [0, 1] or [-50, 50]
                res = (res * (100.0 / 255.0)) - 50.0

                # from numpy array to tensor
                res = torch.from_numpy(res).float()

                # add one channel
                res = res.unsqueeze(0)
                #print(task, res.size())

            elif self.transform is not None and self.transform[task] is not None:
                res = self.transform[task](res)
                #print(task, res.size())


            # Handle special channel tasks
            # base_task = [t for t in SINGLE_IMAGE_TASKS if t == task]
            # if len(base_task) == 0:
            #     continue
            # else:
            #     base_task = base_task[0]
            if 'decoding' in task and res.shape[0] != num_channels:
                assert torch.sum(res[num_channels:,:,:]) < 1e-5, 'unused channels should be 0.'
                res = res[:num_channels,:,:]
            if 'reshading' in task and res.shape[0] != num_channels:
                # res = res[[0],:,:]
                res = res[0].unsqueeze(0)
            #if 'rgb' in task and res.shape[0] != num_channels:

            if self.rgb2lab and 'rgb' in task:
                assert res.shape[0] == 1

            elif 'rgb' in task and res.shape[0] != num_channels:
                if res.shape[0] == 1:
                    res = torch.cat([res] * num_channels, dim=0)
                if res.shape[0] == 4:
                    res = res[:3]
        
            result[task] = res

        return result
    
    def _getitem_single_domain(self, task, building, point, view):
        num_channels = task_parameters[task]['out_channels']
        res = np.uint8(np.zeros((self.image_size, self.image_size, num_channels))).squeeze()
        res = Image.fromarray(res)
        path = self.url_dict[(task, building, point, view)]
        # res = default_loader(path)
        if self.transform is not None and self.transform[task] is not None:
            res = self.transform[task](res)

        # Handle special channel tasks
        # base_task = [t for t in SINGLE_IMAGE_TASKS if t == task]
        # if len(base_task) == 0:
        #     continue
        # else:
        #     base_task = base_task[0]
        num_channels = task_parameters[task]['out_channels']
        if 'decoding' in task and res.shape[0] != num_channels:
            assert torch.sum(res[num_channels:,:,:]) < 1e-5, 'unused channels should be 0.'
            res = res[:num_channels,:,:]
        if 'reshading' in task and res.shape[0] != num_channels:
            # res = res[[0],:,:]
            res = res[0].unsqueeze(0)
        if 'rgb' in task and res.shape[0] != num_channels:
            if res.shape[0] == 1:
                res = torch.cat([res] * num_channels, dim=0)
            if res.shape[0] == 4:
                res = res[:3]
        return res


    def randomize_order(self):
        # random.seed(0)
        random.shuffle(self.bpv_list)
    
    def task_config(self, task):
        return task_parameters[task]

    def _remove_unmatched_images(self) -> (Dict[str, List[str]], int):
        '''
            Filters out point/view/building triplets that are not present for all tasks
            
            Returns:
                filtered_urls: Filtered Dict
                max_length: max([len(urls) for _, urls in filtered_urls.items()])
        '''
        n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
        max_images = max(n_images_task)[0]
        if max(n_images_task)[0] == min(n_images_task)[0]:
            return self.urls, max_images
        else:
            print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))
            # Get views for each task
            def _parse_fpath_for_view( path ):
                building = os.path.basename(os.path.dirname(path))
                file_name = os.path.basename(path) 
                lf = parse_filename( file_name )
                return View(view=lf.view, point=lf.point, building=building)

            self.task_to_view = {}
            for task, paths in self.urls.items():
                self.task_to_view[task] = [_parse_fpath_for_view( path ) for path in paths]
    
            # Compute intersection
            intersection = None
            for task, uuids in self.task_to_view.items():
                if intersection is None:
                    intersection = set(uuids)
                else:
                    intersection = intersection.intersection(uuids)
            # Keep intersection
            print('Keeping intersection: ({} images/task)...'.format(len(intersection)))
            new_urls = {}
            for task, paths in self.urls.items():
                new_urls[task] = [path for path in paths if _parse_fpath_for_view( path ) in intersection]
            return new_urls, len(intersection)
        raise NotImplementedError('Reached the end of this function. You should not be seeing this!')

    def _validate_images_per_building(self):
            # Validate number of images
            print("Building TaskonomyDataset:")
            n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
            print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
            if max(n_images_task)[0] != min(n_images_task)[0]:
                print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                    max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))

                # count number of frames per building per task
                all_building_counts = defaultdict(dict)
                for task, obs in self.urls.items():
                    c = Counter([url.split("/")[-2] for url in obs])
                    for building in c:
                        all_building_counts[building][task] = c[building]

                # find where the number of distinct counts is more than 1
                print('Removing data from the following buildings')
                buildings_to_remove = []
                for b, count in all_building_counts.items():
                    if len(set(list(count.values()))) > 1:
                        print(f"\t{b}:", count)
                        buildings_to_remove.append(b)
                    if len(count) != len(self.tasks):
                        print(f"\t{b}: missing in tasks", set(self.tasks) - set(count.keys()))
                        buildings_to_remove.append(b)
                # [(len(obs), task) for task, obs in self.urls.items()]

                # redo the loading with fewer buildings
                buildings_redo = [b for b in self.buildings if b not in buildings_to_remove]
                self.urls = {task: make_dataset(os.path.join(self.data_path, task), buildings_redo)
                            for task in self.tasks}
                n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
                print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
            assert max(n_images_task)[0] == min(n_images_task)[0], \
                    "Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                    max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task]))
            return n_images_task


def default_file_name_parser(url):
    building = url.split('/')[-2]
    file_name = url.split('/')[-1].split('_')
    point, view = file_name[1], file_name[3]
    return building, point, view

def no_multiview_file_name_parser(url):
    building = url.split('/')[-2]
    file_name = url.split('/')[-1].split('.')[0]
    point, view = file_name, "0"
    return building, point, view

def get_cached_fpaths_filename(tasks, buildings):
    tasks_str = '-'.join(tasks)
    buildings_str = '-'.join(buildings) if isinstance(buildings, list) else buildings
    if len(buildings_str) > 50:
        buildings_str = hashlib.md5(buildings_str.encode('utf-8')).hexdigest()
    return f'./tmp/taskonomy_{tasks_str}_{buildings_str}.pkl'


def make_dataset(dir, folders=None):
    #  folders are building names. If None, get all the images (from both building folders and dir)
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for subfolder in sorted(os.listdir(dir)):
        subfolder_path = os.path.join(dir, subfolder)
        if os.path.isdir(subfolder_path) and (folders is None or subfolder in folders):
            for fname in sorted(os.listdir(subfolder_path)):
                path = os.path.join(subfolder_path, fname)
                images.append(path)

        # If folders/buildings are not specified, use images in dir
        if folders is None and os.path.isfile(subfolder_path):
            images.append(subfolder_path)

    return images

def parse_filename( filename ):
    p = re.match('.*point_(?P<point>\d+)_view_(?P<view>\d+)_domain_(?P<domain>\w+)', filename)
    if p is None:
        raise ValueError( 'Filename "{}" not matched. Must be of form point_XX_view_YY_domain_ZZ.**.'.format(filename) )

    lf = {'point': p.group('point'), 'view': p.group('view'), 'domain': p.group('domain') }
    return LabelFile(**lf)


class TaskonomyDataLoader:

    @dataclass
    class Options(TaskonomyDataset.Options):
        phase: str = 'val'
        batch_size: int = 6
        shuffle: bool = True
        num_workers: int = 8
        pin_memory: bool = True

    def make(options: Options):
        is_train = (options.phase == 'train')
        dataset = TaskonomyDataset(options)
        return data.DataLoader(
            dataset=dataset,
            batch_size=options.batch_size,
            shuffle=options.shuffle,
            num_workers=options.num_workers,
            pin_memory=options.pin_memory,
            drop_last=is_train)
