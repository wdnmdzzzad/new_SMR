import torch
import numpy as np
import math
import time
import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def camera_position_from_spherical_angles(dist, elev, azim, degrees=True):
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.
    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.
    The vectors are broadcast against each other so they all have shape (N, 1).
    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    return camera_position.reshape(-1, 3)


def generate_transformation_matrix(camera_position, look_at, camera_up_direction):
    r"""Generate transformation matrix for given camera parameters.

    Formula is :math:`\text{P_cam} = \text{P_world} * {\text{transformation_mtx}`,
    with :math:`\text{P_world}` being the points coordinates padded with 1.

    Args:
        camera_position (torch.FloatTensor):
            camera positions of shape :math:`(\text{batch_size}, 3)`,
            it means where your cameras are
        look_at (torch.FloatTensor):
            where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
        camera_up_direction (torch.FloatTensor):
            camera up directions of shape :math:`(\text{batch_size}, 3)`,
            it means what are your camera up directions, generally [0, 1, 0]

    Returns:
        (torch.FloatTensor):
            The camera transformation matrix of shape :math:`(\text{batch_size, 4, 3)`.
    """
    z_axis = (camera_position - look_at)
    z_axis = z_axis / z_axis.norm(dim=1, keepdim=True)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis = x_axis / x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = (-camera_position.unsqueeze(1) @ rot_part)
    return torch.cat([rot_part, trans_part], dim=1)



def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(torch.ones_like(d_interpolates)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# def calc_fid():
#     for i, data in tqdm.tqdm(enumerate(test_dataloader)):
#         Xa = Variable(data['data']['images']).cuda()
#         paths = data['data']['path']

#         with torch.no_grad():
#             Ae = netE(Xa)
#             Xer, Ae = diffRender.render(**Ae)

#             Ai = deep_copy(Ae)
#             Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
#             Xir, Ai = diffRender.render(**Ai)

#             for i in range(len(paths)):
#                 path = paths[i]
#                 image_name = os.path.basename(path)
#                 rec_path = os.path.join(rec_dir, image_name)
#                 output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
#                 output_Xer.save(rec_path, 'JPEG', quality=100)

#                 inter_path = os.path.join(inter_dir, image_name)
#                 output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
#                 output_Xir.save(inter_path, 'JPEG', quality=100)

#                 ori_path = os.path.join(ori_dir, image_name)
#                 output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
#                 output_Xa.save(ori_path, 'JPEG', quality=100)
#     fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 32, True)
#     print('Test recon fid: %0.2f' % fid_recon)
#     summary_writer.add_scalar('Test/fid_recon', fid_recon, epoch)

#     fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 32, True)
#     print('Test rotation fid: %0.2f' % fid_inter)
#     summary_writer.add_scalar('Test/fid_inter', fid_inter, epoch)