import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import argparse

sys.path.insert(0, 'video_retalking/third_part')
sys.path.insert(0, 'video_retalking/third_part/GPEN')
sys.path.insert(0, 'video_retalking/third_part/GFPGAN')

# 3dmm extraction
from video_retalking.third_part.face3d.util.preprocess import align_img
from video_retalking.third_part.face3d.util.load_mats import load_lm3d
from video_retalking.third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from video_retalking.third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from video_retalking.third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from video_retalking.third_part.ganimation_replicate.model.ganimation import GANimationModel

from video_retalking.utils import audio
from video_retalking.utils.ffhq_preprocess import Croper
from video_retalking.utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from video_retalking.utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, args, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
from video_retalking.inference import datagen
import warnings
warnings.filterwarnings("ignore")

lipsync_options = args

from talknet_asd.demoTalkNet import frames_asd


def main(video_path,audio_path,output_folder,outfile):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    base_name = os.path.basename(video_path).split(".")[0]
    os.makedirs(os.path.join(output_folder,base_name), exist_ok=True)
    os.makedirs(os.path.join(output_folder,"temp"), exist_ok=True)

    enhancer = FaceEnhancement(base_dir='video_retalking/checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='video_retalking/checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

    asd_output = frames_asd(video_path,output_folder)    
    full_frames = [cv2.imread(os.path.join(output_folder,base_name,"pyframes",im_name)) for im_name in os.listdir(os.path.join(output_folder,base_name,"pyframes")) ]    
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)        
    print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))    
    # crop face of speaking person, in case no person is speaking put full frame
    frames_pil = []
    asd_coordinates = []
    for fidx,frame in enumerate(full_frames,start=0):
        if fidx in asd_output.keys(): # frame is a frame containing a face:
            bbox = asd_output[fidx]["bbox"]
            asd_coordinates.append([int(coordinate) for coordinate in bbox])       
            frames_pil.append(Image.fromarray(cv2.resize(frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])],(256,256))))
        else:
            frames_pil.append(Image.fromarray(cv2.resize(frame,(256,256))))    
    print("\n\n",f"{len(full_frames)}",f"{len(asd_output)}",f"{len(frames_pil)}")    
    
    # get the landmark according to the detected face.
    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+'_landmarks.txt')):
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, os.path.join(output_folder,'temp/',base_name+'_landmarks.txt'))
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt(os.path.join(output_folder,'temp/',base_name+'_landmarks.txt')).astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])
       
    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+'_coeffs.npy')):
        net_recon = load_face3d_net(lipsync_options.face3d_net_path, device)
        lm3d_std = load_lm3d('video_retalking/checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save(os.path.join(output_folder,'temp/',base_name+'_coeffs.npy'), semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load(os.path.join(output_folder,'temp/',base_name+'_coeffs.npy')).astype(np.float32)

    # generate the 3dmm coeff from a single image    
    print('using expression center')
    expression = torch.tensor(loadmat('video_retalking/checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(lipsync_options, device)

    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+'_stablized.npy')):
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):            
            source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
            semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)        
            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device) 
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
        np.save(os.path.join(output_folder,'temp/',base_name+'_stablized.npy'),imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load(os.path.join(output_folder,'temp/',base_name+'_stablized.npy'))
    torch.cuda.empty_cache()

    if not audio_path.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(audio_path, '{}/temp/temp.wav'.format(output_folder))
        subprocess.call(command, shell=True)
        audio_path = '{}/temp/temp.wav'.format(output_folder)
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]  
    lm = lm[:len(mel_chunks)]
    
    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
        imgs_enhanced.append(pred)
    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None,asd_coordinates,output_folder,base_name)

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('{}/temp/result.mp4'.format(output_folder), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    
    if lipsync_options.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    kp_extractor = KeypointExtractor()
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / lipsync_options.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
        
        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1) 
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if lipsync_options.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[lipsync_options.up_face]
            else:
                pass
            
            if lipsync_options.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
                
            if lipsync_options.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                pred = pred * mask + cur_gen_faces * (1 - mask) 
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        torch.cuda.empty_cache()
        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            ff = xf.copy() 
            ff[y1:y2, x1:x2] = p
            
            # month region enhancement by GFPGAN
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                ff, has_aligned=False, only_center_face=True, paste_back=True)
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
            mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mouse_mask = np.zeros_like(restored_img)
            tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
            mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

            height, width = ff.shape[:2]
            restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
            img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
            pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

            pp, orig_faces, enhanced_faces = enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
            out.write(pp)
    out.release()
    
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, '{}/temp/result.mp4'.format(output_folder), outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('outfile:', outfile)

if __name__=="__main__":    
    parser = argparse.ArgumentParser(description = "Lipsync of a speaker in the presence of many faces in the video")
    parser.add_argument('--videoPath', type=str, help='Full path of input video')
    parser.add_argument('--audioPath', type=str, help='Full path for audio to be used for lipsync')
    parser.add_argument('--outputFolder', type=str, help='Full path for the folder to be used for temporary files and output video')
    parser.add_argument('--outFileName', type=str, help='Name of output lipsync video')
    args = parser.parse_args()
    video_path = args.videoPath
    audio_path = args.audioPath
    output_folder = args.outputFolder
    outfile = os.path.join(output_folder,args.outFileName)
    main(video_path,audio_path,output_folder,outfile)