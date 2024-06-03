from rest_framework.decorators import api_view
from rest_framework.response import Response
from .api.serializers import UserSerializer, GeneratedImageSerializer, SegmentedImageSerializer
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from PIL import Image
from .models import GeneratedImage, SegmentedImage
import numpy as np
import cloudinary.uploader
import cloudinary.api
from io import BytesIO
from torchvision.utils import make_grid
import tensorflow as tf
from stylegan2_pytorch import ModelLoader
import torch
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import requests

from rest_framework.exceptions import NotFound

# functions for segformer
def map_label_to_pixel(label):
    pixel_values = [0, 40, 80, 160, 210]
    return pixel_values[label]

def decoded_mask1(one_hot_encoded_mask):
    return torch.tensor([map_label_to_pixel(label) for label in one_hot_encoded_mask.view(-1)])

def replicate_channels(image, num_channels=3):
    return np.repeat(image[:, :, np.newaxis], num_channels, axis=2)

@api_view(['POST'])
def login(request):
    user = get_object_or_404(User, username=request.data['username'])
    if not user.check_password(request.data['password']):
        return Response("missing user")
    token, _ = Token.objects.get_or_create(user=user)
    serializer = UserSerializer(user)
    return Response({'token': token.key, 'user': serializer.data})
@api_view(['POST'])
def signup(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        user = User.objects.get(username=request.data['username'])
        user.set_password(request.data['password'])
        user.save()
        token = Token.objects.create(user=user)
        return Response({'token': token.key, 'user': serializer.data})
    return Response(serializer.errors, status=status.HTTP_200_OK)
@api_view(['GET'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def test_token(request):
    return Response(request.user.username)
@api_view(['POST'])
def getGeneratedImages(request):
    if request.method == 'POST':
        userId = request.data.get('userId')
        if userId is not None:
            try:
                user_instance = User.objects.get(id=userId)
                generated_images = GeneratedImage.objects.filter(userId=user_instance)
                serialized_data = GeneratedImageSerializer(generated_images, many=True).data
                return Response({"generated_images": serialized_data}, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({"message": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({"message": "userId parameter is missing"}, status=status.HTTP_400_BAD_REQUEST)
    return Response({"message": "Hello, world!!!"})
@api_view(['POST'])
def getSegmentedImages(request):
    if request.method == 'POST':
        userId = request.data.get('userId')
        if userId is not None:
            try:
                user_instance = User.objects.get(id=userId)
                if not user_instance:
                    return Response({ "st": "message" })
                generated_images = SegmentedImage.objects.filter(userId=user_instance)
                # if not generated_images:
                #     return Response({ "s": "message" })
                serialized_data = SegmentedImageSerializer(generated_images, many=True).data
                return Response({"segmented_images": serialized_data}, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({"message": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({"message": "userId parameter is missing"}, status=status.HTTP_400_BAD_REQUEST)
    return Response({"message": "Hello, world!!!"})

#unet model
@api_view(['POST'])
def createImages(request):
    if request.method == 'POST':
        user_id = request.data.get('userId')
        user = User.objects.filter(id=user_id).first()
        if not user:
            raise NotFound("User not found")
        threshold = 0.5
        model_1 = tf.keras.models.load_model('./savedModels/Segemntationmodel.hdf5')
        
        # img_path =  request.data.get('urll')
        img_path = request.data.get('urll')
        orig_img = Image.open(BytesIO(requests.get(img_path).content)).convert('L')
        # orig_img = Image.open(img_path)
        orig_img = orig_img.resize((512, 496))
        orig_img = np.array(orig_img)
        orig_img =np.expand_dims(orig_img, 0)
        mask_1 = (model_1.predict(orig_img)[0,:,:,0] < threshold).astype(np.uint8)
        mask_1 = mask_1 * 255
        mask = np.where(mask_1 > 128, 1, 0).astype(np.uint8)
        orig_img = np.squeeze(orig_img)
        orig_img = np.multiply(orig_img,mask)
        orig_img = Image.fromarray(orig_img)
        orig_img = orig_img.resize((512, 512)).convert('RGB')
        if orig_img:
            img_bytes = BytesIO()
            orig_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            cloudinary_response = cloudinary.uploader.upload(img_bytes)
            cloudinary_url = cloudinary_response.get("secure_url")
            generated_image_instance = GeneratedImage.objects.create(userId=user, link=cloudinary_url, type=5, generated=0)
            generated_image_instance.save()
            return Response({"message": "Mission Completed!"})
    return Response({"message": "Mission Failed!"})

# GAN new Experiment with server
def generate(user, num,diease):
    if not user:
        raise NotFound("User not found")
    if num == 0:
        print("No images to generate")
        return
    if diease == 0:
        loader = ModelLoader(base_dir='./savedModels/Normal', name='default')
    elif diease == 1:
        loader = ModelLoader(base_dir='./savedModels/CNV', name='default')
    elif diease == 2:
        loader = ModelLoader(base_dir='./savedModels/DME', name='default')
    elif diease == 3:
        loader = ModelLoader(base_dir='./savedModels/Drusen', name='default')
    # this is the number of images you want to generate user wants to put in the database
    t_images = num
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    for _ in range(t_images):
        noise = torch.randn(1, 512).to(device)
        styles = loader.noise_to_styles(noise=noise, trunc_psi=0.9)
        images = loader.styles_to_images(styles)
        grid = make_grid(images[0])
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        cloudinary_response = cloudinary.uploader.upload(img_bytes)
        cloudinary_url = cloudinary_response.get("secure_url")
        generated_image_instance = GeneratedImage.objects.create(userId=user, link=cloudinary_url, type=0, generated=1)
        generated_image_instance.save()
    
    print(f'{num} images generated for diease {diease}')
#gan model
@api_view(['POST'])
def generateImages(request):
    userId = request.data.get('userId')
    normal = request.data.get('normal')
    cnv = request.data.get('cnv')
    dme = request.data.get('dme')
    drusen = request.data.get('drusen')
    user_instance = User.objects.get(id=userId)
    generate(user_instance, normal, diease=0)
    generate(user_instance, cnv,    diease=1)
    generate(user_instance, dme,    diease=2)
    generate(user_instance, drusen, diease=3)
    return Response({ "Mission Completed" })
#segformer model
@api_view(['POST'])
def segmentImages(request):
    if request.method == 'POST':
        user_id = request.data.get('userId')
        generated_image_id = request.data.get('generatedImageId')
        user = User.objects.filter(id=user_id).first()
        generated_image = get_object_or_404(GeneratedImage, id=generated_image_id)
        if not user:
            raise NotFound("User not found")
        if not generated_image:
            raise NotFound("User not found")
        img_path = generated_image.link
        image = Image.open(BytesIO(requests.get(img_path).content)).convert('RGB')
        image = image.resize((512, 496))
        image = np.array(image)
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_extractor = SegformerFeatureExtractor()
        model_state_dict = torch.load('./savedModels/your_model_epoch_120(2).pt', map_location=device)
        mymodel = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=5)
        mymodel = mymodel.to(device)
        mymodel.load_state_dict(model_state_dict)
        mymodel.eval()
        encodings = feature_extractor(image, return_tensors="pt")
        pixel_values = encodings.pixel_values.to(device)
        output = mymodel(pixel_values=pixel_values)
        logits = output.logits.cuda()
        rescaled_logit = torch.nn.functional.interpolate(
            logits,
            size=image[:, :, -1].shape,
            mode='bilinear',
            align_corners=False
        )
        seg_msk = rescaled_logit.argmax(1)[0]
        seg_msk.squeeze().unique()
        color_seg = decoded_mask1(seg_msk)
        color_seg = color_seg.view(496, 512)
        # color_seg = color_seg.resize(512, 512)
        if True:
            img_bytes = BytesIO()
            img = np.array(color_seg)
            image = Image.fromarray(img.astype('uint8'))
            image.save(img_bytes, format='PNG')
            # color_seg.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            cloudinary_response = cloudinary.uploader.upload(img_bytes)
            cloudinary_url = cloudinary_response.get("secure_url")
            segmented_image_instance = SegmentedImage.objects.create(userId=user, generatedImageId=generated_image, link=cloudinary_url, type=1)
            segmented_image_instance.save()
        return Response({"message": "Mission Completed!"})
    return Response({"message": "Mission Failed!"})# @api_view(['POST'])
