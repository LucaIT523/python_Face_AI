import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from reflib.face.restoration import CFaceRestoration
from reflib.face.utils import CFaceUtils


g_deviceInfo = 'cuda'
g_bInit = False

def inpainting(in_image_path, out_image_path):
    global g_bInit
    global g_deviceInfo

    if(g_bInit == False):
        #. registry, model init
        network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=512, conlist=['32', '64', '128']).to(g_deviceInfo)
        model_path = 'reflib/model/inpainting.pth'
        model_H = torch.load(model_path)['params_ema']
        network.load_state_dict(model_H)
        network.eval()
        g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tInpainting start ... {fileName}')

    # image size(512 * 512)
    infaceImg = cv2.imread(in_image_path)
    assert infaceImg.shape[:2] == (512, 512), 'The resolution of image must be 512x512.'
    infaceImg = img2tensor(infaceImg / 255., bgr2rgb=True, float32=True)
    normalize(infaceImg, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    infaceImg = infaceImg.unsqueeze(0).to(g_deviceInfo)

    try:
        with torch.no_grad():
            mask = torch.zeros(512, 512)
            mask_id = torch.sum(infaceImg[0], dim=0)
            mask[mask_id == 3] = 1.0
            mask = mask.view(1, 1, 512, 512).to(g_deviceInfo)

            # makse_img = tensor2img(mask, rgb2bgr=True, min_max=(-1, 1))
            outfaceImg = network(infaceImg, 1, False)[0]
            outfaceImg = (1 - mask) * infaceImg + mask * outfaceImg
            savefaceImg = tensor2img(outfaceImg, rgb2bgr=True, min_max=(-1, 1))

        del outfaceImg
        torch.cuda.empty_cache()
        # Save
        savefaceImg = savefaceImg.astype('uint8')
        imwrite(savefaceImg, out_image_path)
        print(f'\tInpainting OK ... {out_image_path}')

    except Exception as error:
        print(f'\tFailed inpainting: {error}')




def colorization(in_image_path, out_image_path):
    global g_bInit
    global g_deviceInfo

    if(g_bInit == False):
        #. registry, model init
        network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=1024, conlist=['32', '64', '128']).to(g_deviceInfo)
        model_path = 'reflib/model/colorization.pth'
        model_H = torch.load(model_path)['params_ema']
        network.load_state_dict(model_H)
        network.eval()
        g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tColorization start ... {fileName}')

    # image size(512 * 512)
    infaceImg = cv2.imread(in_image_path)
    assert infaceImg.shape[:2] == (512, 512), 'The resolution of image must be 512x512.'
    infaceImg = img2tensor(infaceImg / 255., bgr2rgb=True, float32=True)
    normalize(infaceImg, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    infaceImg = infaceImg.unsqueeze(0).to(g_deviceInfo)

    try:
        with torch.no_grad():

            outfaceImg = network(infaceImg, 0, False)[0]
            savefaceImg = tensor2img(outfaceImg, rgb2bgr=True, min_max=(-1, 1))

        del outfaceImg
        torch.cuda.empty_cache()
        # Save
        savefaceImg = savefaceImg.astype('uint8')
        imwrite(savefaceImg, out_image_path)
        print(f'\tColorization OK ... {out_image_path}')

    except Exception as error:
        print(f'\tFailed colorization: {error}')

# The unoptimized RealESRGAN is slow on CPU.
# def set_realESRGan():
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#     from basicsr.utils.realesrgan_utils import RealESRGANer
#
#     use_half = False
#     if torch.cuda.is_available(): # set False in CPU/MPS mode
#         no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
#         if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
#             use_half = True
#
#     model = RRDBNet(
#                 num_in_ch=3,
#                 num_out_ch=3,
#                 num_feat=64,
#                 num_block=23,
#                 num_grow_ch=32,
#                 scale=2,
#     )
#     upsampler = RealESRGANer(
#                     scale=2,
#                     model_path="reflib/model/RealESRGAN_x2plus.pth",
#                     model=model,
#                     tile=args.bg_tile,
#                     tile_pad=40,
#                     pre_pad=0,
#                     half=use_half
#     )
#     return upsampler


def restoration(in_image_path, out_image_path, face_temp_path):
    global g_bInit
    global g_deviceInfo

    if (g_bInit == False):
        # . registry, model init
        network = ARCH_REGISTRY.get('FaceSuper')(dim=512, cbSize=1024, conlist=['32', '64', '128', '256']).to(g_deviceInfo)
        model_path = 'reflib/model/restoration.pth'
        model_H = torch.load(model_path)['params_ema']
        network.load_state_dict(model_H)
        network.eval()
        g_bInit = True

    fileName = os.path.basename(in_image_path)
    print(f'\tRestoration start ... {fileName}')

    faceRestor = CFaceRestoration(  upscale_factor= 2,  # Default : 2
                                    face_size=512,
                                    crop_ratio=(1, 1),
                                    det_model = 'retinaface_resnet50',
                                    save_ext='png',
                                    use_parse=True,
                                    device=g_deviceInfo)

    # isinstance(in_image_path, str)
    img = cv2.imread(in_image_path, cv2.IMREAD_COLOR)
    # faceRestor.init()
    faceRestor.read_image(img)
    # get landmark of face
    detFaces = faceRestor.get_face_landmarks_5(False, False, 640, eye_dist_threshold=5)
    print(f'\tNumber of face detection - ({detFaces} faces)')
    faceRestor.align_warp_face()

    for _, idx_face in enumerate(faceRestor.cropped_faces):

        idx_face_tensor = img2tensor(idx_face / 255., True, True)
        normalize(idx_face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        idx_face_tensor = idx_face_tensor.unsqueeze(0).to(g_deviceInfo)

        try:
            with torch.no_grad():
                res_tensor = network(idx_face_tensor, 0.5, True)[0]
                restored_face = tensor2img(res_tensor, True, min_max=(-1, 1))
            del res_tensor
            torch.cuda.empty_cache()

            restored_face = restored_face.astype('uint8')
            faceRestor.add_restored_face(restored_face, idx_face)

        except Exception as error:
            print(f'\tFailed restoration: {error}')

    faceRestor.get_inverse_affine(None)
    savefaceImg = faceRestor.paste_orgimage(upsample_img=None, draw_box=False)
    for idx, (_, restored_face) in enumerate(zip(faceRestor.cropped_faces, faceRestor.restored_faces)):
        fileName = os.path.basename(in_image_path)
        face_image_path = f'{face_temp_path}{fileName}_{idx}.png'
        imwrite(restored_face, face_image_path)
    # Save
    imwrite(savefaceImg, out_image_path)
    print(f'\tRestoration OK ... {out_image_path}')

if __name__ == '__main__':

    g_deviceInfo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # check argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./input', help='Input image or folder')
    parser.add_argument('-option', type=str, default='inpaint',  help='color - face image colorzation, inpaint - face image inpainting, restor - face image restoration(cropped)')
    parser.add_argument('-result', type=str, default='./result',  help='Output folder')
    args = parser.parse_args()

    # file
    if args.input.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        fileName = os.path.basename(args.input)
        if(args.option == 'inpaint'):
            out_imgPath = f'{args.result}/inpainting/{fileName}'
            inpainting(args.input, out_imgPath)
        elif(args.option == 'color'):
            out_imgPath = f'{args.result}/colorization/{fileName}'
            colorization(args.input, out_imgPath)
        elif (args.option == 'restor'):
            out_imgPath = f'{args.result}/restoration/{fileName}'
            out_FaceTempPath = f'{args.result}/restoration/face/'
            restoration(args.input, out_imgPath, out_FaceTempPath)
        else:
            print(f'\tError ... -option param')

    # folder
    elif args.input.endswith('/'):
        # scan
        input_img_list = sorted(glob.glob(os.path.join(args.input, '*.[jpJP][pnPN]*[gG]')))
        fileName = os.path.basename(args.input)

    else:
        print(f'\tError ... param')



