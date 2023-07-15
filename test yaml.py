# import omegaconf
# import yaml
# import os
# import json
#
# state = flax.jax_utils.replicate(model_ckpt['model'])
# time_steps = [20, 25, 35, 50, 75, 100, 200, 250, 500, 1000]
# for time in time_steps:
#     c.sampling_timesteps = time
#     sample_save_image(key, c, time, state)

# def pyramid_noise_like(x, discount=0.9):
#   b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
#   u = nn.Upsample(size=(w, h), mode='bilinear')
#   noise = torch.randn_like(x)
#   for i in range(10):
#     r = random.random()*2+2 # Rather than always going 2x,
#     w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
#     noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
#     if w==1 or h==1: break # Lowest resolution is 1x1
#   return noise/noise.std() # Scaled back to roughly unit variance

#


import numpy as np
import cv2


if __name__=="__main__":


    def add_gaussian_noise(image, target_snr):
        # Convert image to grayscale
        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image =image

        # Calculate signal power
        signal_power = np.mean(gray_image) ** 2

        # Calculate noise power
        noise_power = signal_power / (10 ** (target_snr / 10))

        # Generate Gaussian noise
        noise = np.random.normal(0, 1, gray_image.shape)

        # Calculate generated noise power
        generated_noise_power = np.mean(noise) ** 2

        # Scale the noise to match the target noise power
        scaled_noise = noise * np.sqrt(noise_power / generated_noise_power)

        # Add scaled noise to the original image
        noisy_image = gray_image + scaled_noise

        # Clip the values to the valid range of [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)

        # Convert the image back to uint8
        noisy_image = noisy_image.astype(np.uint8)

        return noisy_image


    # 读取原始图像
    image = cv2.imread('/home/john/ダウンロード/alice.png')

    # 添加高斯噪声并指定信噪比为20dB
    noisy_image = add_gaussian_noise(image, 60)

    # 保存添加噪声后的图像
    cv2.imwrite('noisy_image.jpg', noisy_image)
