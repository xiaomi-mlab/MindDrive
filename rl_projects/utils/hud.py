import cv2
import os
import numpy as np
import pickle
import pygame
from loguru import logger
from ..agents.utils import format_results
from ..utils.render import Renderer

class HUD(object):
    def __init__(self, cfg, render_state, render_expert):
        self.cfg = cfg
        self.render_state = render_state
        self.render_expert = render_expert
        render_cfg = {
            'render_state' : render_state,
            'render_expert' : render_expert
        }
        self.render_ = Renderer(grid_config=self.cfg.grid_config, render_cfg=render_cfg)
        logger.info('render vis dir : {}'.format(self.cfg.vis_dir))
        os.makedirs(self.cfg.vis_dir, exist_ok=True)
        pygame.init()
        pygame.font.init()
        self.dim = (3690, 2160)
        self.display = pygame.display.set_mode(self.dim, pygame.HWSURFACE | pygame.DOUBLEBUF)
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font_mono = pygame.font.Font(mono, 50)

        self.mask_width = 960

    def render(self, new_obs, predictions, infos):
        ## vis batch 0
        extrinsics = new_obs['extrinsics'][0]
        intrinsics = new_obs['intrinsics'][0]
        render_imgs = new_obs['render_camera'][0]
        timestamp = float(new_obs['timestamp'][0][0])
        clip_name = infos[0]['clip_name']
        objs, lanes = format_results(predictions['perception_res'], self.cfg)
        waypoints = infos[0]['expert_traj']
        save_img = self.render_.render_imgs_and_annos(render_imgs, objs, lanes, waypoints, extrinsics, intrinsics)

        self.display.blit(pygame.surfarray.make_surface(save_img.swapaxes(0, 1)), (0, 0))

        info_text = [
                    "Scene name: {}".format(clip_name),
                    "",
                    "Timestamp:   % 4s" % (str(timestamp)),
                    "",
                    "Lat Speed:   % 5.0f km/h" % (3.6 * infos[0]['ego_velocity'][0]),
                    "Lon Speed:   % 5.0f km/h" % (3.6 * infos[0]['ego_velocity'][1]),
                    "",
                    "Metric------------------------------",
                    "Distance traveled: %5.0f m" % (infos[0]['total_distance']),
                    "Collision per meters: %5.0f m" % (infos[0]['CPM']),
                    "",
                    "Collision with Object: %5s" % (infos[0]['coll_obj_type']),
                    "Collision with Lane: %5s" % (infos[0]['coll_lane_type']),
                    # ("Throttle:", throttle, 0.0, 1.0),
                    # ("Steer:", steer, -1.0, 1.0),
                    # ("Brake:", brake, 0.0, 1.0)
                ]

        info_surface = pygame.Surface((self.mask_width, self.dim[1]))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))
        v_offset = 4
        v_diff_offset = 50
        # bar_h_offset = self.dim[0] - self.mask_width + 20
        bar_h_offset = 10
        bar_width = 106

        for item in info_text:
            if v_offset + v_diff_offset > self.dim[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(self.display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += v_diff_offset
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect)
                item = item[0]
            if item: # At this point has to be a str.
                surface = self.font_mono.render(item, True, (0, 0, 0))
                self.display.blit(surface, (bar_h_offset, v_offset))
            v_offset += v_diff_offset

        save_img = np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        save_dir = f'{self.cfg.vis_dir}/{clip_name}'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/{timestamp}_pred.jpg', save_img)
