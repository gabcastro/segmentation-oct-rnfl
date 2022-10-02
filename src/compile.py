import tensorflow as tf
from tensorflow.keras import backend as K

class Compile:
    """Compute all metrics and losses used during fit model"""
    
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss = self.dice_coef_loss
        self.all_metrics = [self.dice_coef, self.soft_dice_coef, self.iou_coef]

    def dice_coef_loss(self, target, prediction, axis=(1, 2), smooth=1):
        """
            Sorenson (Soft) Dice loss
            Using -log(Dice) as the loss since it is better behaved.
            Also, the log allows avoidance of the division which
            can help prevent underflow when the numbers are very small.
        """
        intersection = tf.reduce_sum(prediction * target, axis=axis)
        p = tf.reduce_sum(prediction, axis=axis)
        t = tf.reduce_sum(target, axis=axis)
        numerator = tf.reduce_mean(intersection + smooth)
        denominator = tf.reduce_mean(t + p + smooth)
        dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

        return dice_loss

    def dice_coef(self, target, prediction, axis=(1, 2), smooth=1):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = K.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def soft_dice_coef(self, target, prediction, axis=(1, 2), smooth=1):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def iou_coef(self, y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou