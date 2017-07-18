# Needs Python 2.7 and Keras 1.2.2
#
# To set this up:
# virtualenv -p /usr/bin/python2.7 env
# source env/bin/activate
# pip install tensorflow
# pip install keras==1.2.2
# pip install coremltools
#
# Use "deactivate" when you're done.

import coremltools

coreml_model = coremltools.converters.caffe.convert(
    ('EmotiW_VGG_S.caffemodel', 'EmotiW_VGG_S.prototxt'),
    image_input_names='data',
    class_labels='labels.txt')

coreml_model.author = 'Gil Levi and Tal Hassner, Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns, Proc. ACM International Conference on Multimodal Interaction (ICMI), Seattle, Nov. 2015'
coreml_model.license = 'Unknown'
coreml_model.short_description = "Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns"
coreml_model.input_description['data'] = 'Input image to be classified'
coreml_model.output_description['prob'] = 'Probability of each emotion'
coreml_model.output_description['classLabel'] = 'Most likely image emotion'

print(coreml_model)

# Test that the converted network gives the same output as the original
# model. The top 5 predictions should be:
#   0.29618 n02123159 tiger cat 282
#   0.14749 n02119022 red fox, Vulpes vulpes 277
#   0.13466 n02119789 kit fox, Vulpes macrotis 278
#   0.08651 n02113023 Pembroke, Pembroke Welsh corgi 263
#   0.03148 n02123045 tabby, tabby cat 281
#
# To run this you need macOS 10.13 and the following packages:
#   pip install pillow

#from PIL import Image
#cat = Image.open('../cat.jpg')
#print(coreml_model.predict({'data': cat}))

coreml_model.save('EmotiW_VGG_S.mlmodel')
