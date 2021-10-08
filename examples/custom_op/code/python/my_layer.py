import caffe

class MyAdd(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data = bottom[0].data + bottom[1].data

    def backward(self, top, propagate_down, bottom):
        pass

class MyMul(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data = bottom[0].data * bottom[1].data

    def backward(self, top, propagate_down, bottom):
        pass