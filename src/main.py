from unet import Unet
import tensorflow as tf

def main():
    input = tf.keras.Input(shape=(512, 512, 1))
    unet = Unet()
    unet(input)
    unet.summary()

if __name__ == "__main__":
    main()