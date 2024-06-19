import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import to_categorical
from model import CNN1, CNN2, CNN3,optimized_CNN, optimized_CNN_with_se, optimized_CNN_with_cbam
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow_addons.optimizers import RectifiedAdam
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def load_data(data_path, categories, img_size=(48, 48)):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_path, category)
        for img_name in os.listdir(path):
            if img_name.endswith('.png'):
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).resize(img_size).convert('L')
                img = np.array(img)
                data.append(img)
                labels.append(category)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# 定义生成 Grad-CAM 的函数
def generate_grad_cam(model, img, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = np.uint8(cam * 255)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    return cam

# 定义可视化 Grad-CAM 的函数
def plot_grad_cam(model, img, last_conv_layer_name):
    cam = generate_grad_cam(model, img, last_conv_layer_name)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img)
    plt.axis('off')
    # plt.show()
    plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# 预处理图像
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # 添加通道维度
    return img

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    return None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--model', default="optimized_CNN", type=str)
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--img_path', default="./data/test/angry/im0.png", type=str)
    
    args = parser.parse_args()

    categories = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    train_and_val_data, train_and_val_labels = load_data('./data/train', categories)
    test_data, test_labels = load_data('./data/test', categories)

    train_and_val_data = train_and_val_data / 255.0
    test_data = test_data / 255.0

    train_and_val_data = train_and_val_data.reshape(-1, 48, 48, 1)
    test_data = test_data.reshape(-1, 48, 48, 1)

    le = LabelEncoder()
    train_and_val_labels = to_categorical(le.fit_transform(train_and_val_labels), num_classes=7)
    test_labels = to_categorical(le.transform(test_labels), num_classes=7)

    train_data, val_data, train_labels, val_labels = train_test_split(train_and_val_data, train_and_val_labels, test_size=0.2, random_state=42)

    if args.optimizer == "Adam":
        optimizer = Adam()
    elif args.optimizer == "AdamW":
        optimizer = AdamW(learning_rate=args.lr)
    elif args.optimizer == "RectifiedAdam":
        optimizer = RectifiedAdam(learning_rate=args.lr)
    else:
        raise ValueError('Unknown optimizer')

    if args.model == "CNN1":
        model = CNN1(input_shape=(48, 48, 1), n_classes=7)
    elif args.model == "CNN2":
        model = CNN2(input_shape=(48, 48, 1), n_classes=7)
    elif args.model == "CNN3":
        model = CNN3(input_shape=(48, 48, 1), n_classes=7)
    elif args.model == "optimized_CNN":
        model = optimized_CNN(input_shape=(48, 48, 1), n_classes=7)
    elif args.model == "optimized_CNN_with_se":
        model = optimized_CNN_with_se(input_shape=(48, 48, 1), n_classes=7)
    elif args.model == "optimized_CNN_with_cbam":
        model = optimized_CNN_with_cbam(input_shape=(48, 48, 1), n_classes=7)
    else:
        raise ValueError('Unknown model')

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=args.epoch, batch_size=args.batch_size, validation_data=(val_data, val_labels))

    optimized_score = model.evaluate(test_data, test_labels, verbose=0)
    print(f'Test accuracy: {optimized_score[1] * 100:.2f}%')

    # 示例图像路径
    image_path = args.img_path
    img = preprocess_image(image_path)

    # 可视化 Grad-CAM
    last_conv_layer_name = get_last_conv_layer_name(model)
    plot_grad_cam(model, img, last_conv_layer_name)
