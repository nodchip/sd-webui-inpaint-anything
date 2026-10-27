import ast
import gc
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import types
import unittest

import cv2
import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo
import torch


root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / 'scripts/inpaint_anything.py').read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('auto_resize_to_pil', 'run_inpaint', 'run_cn_inpaint', 'run_webui_inpaint')]
for function in functions:
    function.decorator_list = []


class InpaintMaskTests(unittest.TestCase):
    def test_final_editor_mask_reaches_inpainting_pipeline(self):
        image = np.full((8, 8, 3), 100, np.uint8)
        mask = np.zeros_like(image)
        mask[1:4, 2:5] = 255
        captured = {}

        class Pipeline:
            scheduler = types.SimpleNamespace(config={})
            def to(self, device):
                return self
            def enable_attention_slicing(self):
                pass
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(images=[kwargs['image']])

        scheduler = types.SimpleNamespace(from_config=lambda config: None)
        with tempfile.TemporaryDirectory() as output:
            env = {'np': np, 'cv2': cv2, 'Image': Image, 'ImageFilter': ImageFilter, 'PngInfo': PngInfo,
                   'torch': torch, 'platform': platform, 'random': random, 'gc': gc, 'os': os,
                   'sam_dict': {'mask_image': mask}, 'set_ia_config': lambda *a: None,
                   'IAConfig': types.SimpleNamespace(KEYS=types.SimpleNamespace(INP_MODEL_ID='id'),
                                                     SECTIONS=types.SimpleNamespace(USER='user')),
                   'save_mask_image': lambda *a: None, 'get_webui_setting': lambda *a: False,
                   'download_model_from_hf': lambda *a, **kw: 'available',
                   'IAFileManager': types.SimpleNamespace(DOWNLOAD_COMPLETE='available'),
                   'ia_logging': types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None),
                   'devices': types.SimpleNamespace(cpu=torch.device('cpu'), device=torch.device('cpu')),
                   'ia_check_versions': types.SimpleNamespace(torch_on_amd_rocm=False,
                                                              diffusers_enable_cpu_offload=False),
                   'shared': types.SimpleNamespace(xformers_available=False),
                   'StableDiffusionInpaintPipeline': types.SimpleNamespace(from_pretrained=lambda *a, **kw: Pipeline()),
                   'DDIMScheduler': scheduler,
                   'ia_file_manager': types.SimpleNamespace(outputs_dir=output, savename_prefix='test')}
            exec(compile(ast.Module(body=functions, type_ignores=[]), '<inpaint>', 'exec'), env)
            results = list(env['run_inpaint'](image, True, 'prompt', '', 1, 1, 42, 'model', False, False))
        self.assertEqual(len(results), 1)
        np.testing.assert_array_equal(np.asarray(captured['mask_image']), mask)
        self.assertEqual(captured['prompt'], 'prompt')

    def test_controlnet_and_webui_receive_current_mask(self):
        image = np.full((8, 8, 3), 100, np.uint8)
        mask = np.zeros_like(image)
        mask[1:4, 2:5] = 255
        captured = {}
        processing = types.SimpleNamespace(scripts=object())
        cnet = types.SimpleNamespace(ResizeMode=types.SimpleNamespace(RESIZE='resize'),
                                     to_processing_unit=lambda value: value,
                                     update_cn_script_in_processing=lambda p, units: captured.update(unit=units[0]))
        def create_processing(init_image, mask_image, *args):
            captured['processing_mask'] = np.array(mask_image)
            return processing
        noop = lambda *args, **kwargs: None
        env = {'np': np, 'cv2': cv2, 'Image': Image, 'os': os, 'gc': gc, 'random': random, 're': re,
               'sam_dict': {'mask_image': mask, 'cnet': cnet},
               'shared': types.SimpleNamespace(sd_model=types.SimpleNamespace(parameterization='eps', is_sdxl=False),
                                               opts=types.SimpleNamespace(sd_vae='')),
               'ia_logging': types.SimpleNamespace(info=noop, warning=noop, error=noop),
               'await_pre_reload_model_weights': noop, 'await_backup_reload_ckpt_info': noop,
               'get_closet_checkpoint_match': lambda *args: object(),
               'set_ia_config': noop, 'IAConfig': types.SimpleNamespace(
                   KEYS=types.SimpleNamespace(INP_WEBUI_MODEL_ID='id'), SECTIONS=types.SimpleNamespace(USER='user')),
               'save_mask_image': noop, 'get_sd_img2img_processing': create_processing,
               'backup_alwayson_scripts': noop, 'disable_alwayson_scripts_wo_cn': noop,
               'disable_all_alwayson_scripts': noop, 'restore_alwayson_scripts': noop,
               'clear_controlnet_cache': noop, 'get_controlnet_args_to': lambda *args: 0,
               'get_max_args_to': lambda *args: 0, 'process_images': lambda p: None,
               'devices': types.SimpleNamespace(NansException=type('NansException', (Exception,), {})),
               'ia_check_versions': types.SimpleNamespace(webui_refiner_is_available=False)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<inpaint>', 'exec'), env)
        list(env['run_cn_inpaint'](image, True, 'prompt', '', 'sampler', 1, 1, 1, 42,
                                   'inpaint', 'control', False, False, 1, 'Balanced'))
        np.testing.assert_array_equal(captured['processing_mask'], mask)
        np.testing.assert_array_equal(captured['unit']['image']['mask'], mask)
        captured.clear()
        list(env['run_webui_inpaint'](image, True, 'prompt', '', 'sampler', 1, 1, 1, 42,
                                      'model', False, 4, 0))
        np.testing.assert_array_equal(captured['processing_mask'], mask)


if __name__ == '__main__':
    unittest.main()
