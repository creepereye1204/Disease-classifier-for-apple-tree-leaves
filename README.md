# Disease classifier for apple tree leaves
사과나무 잎의 질병 분류기입니다.

# result.zip에는 loss와 Acc의 결과가 있습니다.



![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/aeff8f9b-4e3e-485c-afd8-4b08a0fc728f)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/de65f4ba-8e55-442e-a41a-cb328599397a)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/38796d3d-7869-401e-8c52-ff9cdce65d09)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/b74a4f0a-fa26-469b-aa34-3500b3e8d8b8)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/5025f47b-4489-4bee-9fa4-3395975a0636)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/a9dc629e-5c07-45cd-a67e-abac6d73e3b0)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/5285f7c9-c30c-4d6a-b674-9254cb8f0238)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/169118a2-2acf-4898-b039-02392d069439)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/aa1fde0c-be31-4c21-92d4-031e5cb30edb)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/a717cea6-e97f-48ca-97f1-dcc2ddc3767a)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/179ffb45-46da-40a5-b96e-d91f80152613)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/e84e74f5-9077-42b6-8924-32721183174f)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/2e35d52f-22e0-4a7d-97ac-5542e7fe6c37)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/c998b100-5795-4955-8bf8-dc9cfa82a58b)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/4ab167a4-1713-491e-b9ca-93f2a761b5f2)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/efc3229c-4aaf-4bd7-acaa-6ea0ad0cb8d9)
![image](https://github.com/creepereye1204/Disease-classifier-for-apple-tree-leaves/assets/112455232/f71a24f4-b4e5-4284-86f1-6abceac3bf47)










```python
!pip install -U scikit-learn
```

    Collecting scikit-learn
      Downloading scikit_learn-1.2.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.8/9.8 MB[0m [31m50.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.8/site-packages (from scikit-learn) (1.23.5)
    Collecting threadpoolctl>=2.0.0
      Downloading threadpoolctl-3.1.0-py3-none-any.whl (14 kB)
    Collecting joblib>=1.1.1
      Downloading joblib-1.2.0-py3-none-any.whl (297 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m298.0/298.0 KB[0m [31m24.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.8/site-packages (from scikit-learn) (1.10.1)
    Installing collected packages: threadpoolctl, joblib, scikit-learn
    Successfully installed joblib-1.2.0 scikit-learn-1.2.2 threadpoolctl-3.1.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: You are using pip version 22.0.4; however, version 23.1.2 is available.
    You should consider upgrading via the '/usr/local/bin/python -m pip install --upgrade pip' command.[0m[33m
    [0m


```python
!pip install  plotly
```

    Collecting plotly
      Downloading plotly-5.15.0-py2.py3-none-any.whl (15.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.5/15.5 MB[0m [31m66.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.8/site-packages (from plotly) (23.0)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.8/site-packages (from plotly) (8.2.2)
    Installing collected packages: plotly
    Successfully installed plotly-5.15.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: You are using pip version 22.0.4; however, version 23.1.2 is available.
    You should consider upgrading via the '/usr/local/bin/python -m pip install --upgrade pip' command.[0m[33m
    [0m


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import tensorflow.keras.optimizers.schedules as schedules
```

    D0610 03:52:21.196883462      14 config.cc:119]                        gRPC EXPERIMENT tcp_frame_size_tuning               OFF (default:OFF)
    D0610 03:52:21.196928195      14 config.cc:119]                        gRPC EXPERIMENT tcp_rcv_lowat                       OFF (default:OFF)
    D0610 03:52:21.196931984      14 config.cc:119]                        gRPC EXPERIMENT peer_state_based_framing            OFF (default:OFF)
    D0610 03:52:21.196934745      14 config.cc:119]                        gRPC EXPERIMENT flow_control_fixes                  ON  (default:ON)
    D0610 03:52:21.196937162      14 config.cc:119]                        gRPC EXPERIMENT memory_pressure_controller          OFF (default:OFF)
    D0610 03:52:21.196939865      14 config.cc:119]                        gRPC EXPERIMENT unconstrained_max_quota_buffer_size OFF (default:OFF)
    D0610 03:52:21.196942410      14 config.cc:119]                        gRPC EXPERIMENT new_hpack_huffman_decoder           ON  (default:ON)
    D0610 03:52:21.196951627      14 config.cc:119]                        gRPC EXPERIMENT event_engine_client                 OFF (default:OFF)
    D0610 03:52:21.196954241      14 config.cc:119]                        gRPC EXPERIMENT monitoring_experiment               ON  (default:ON)
    D0610 03:52:21.196957129      14 config.cc:119]                        gRPC EXPERIMENT promise_based_client_call           OFF (default:OFF)
    D0610 03:52:21.196959605      14 config.cc:119]                        gRPC EXPERIMENT free_large_allocator                OFF (default:OFF)
    D0610 03:52:21.196962163      14 config.cc:119]                        gRPC EXPERIMENT promise_based_server_call           OFF (default:OFF)
    D0610 03:52:21.196964585      14 config.cc:119]                        gRPC EXPERIMENT transport_supplies_client_latency   OFF (default:OFF)
    D0610 03:52:21.196966897      14 config.cc:119]                        gRPC EXPERIMENT event_engine_listener               OFF (default:OFF)
    I0610 03:52:21.197149996      14 ev_epoll1_linux.cc:122]               grpc epoll fd: 62
    D0610 03:52:21.205923103      14 ev_posix.cc:144]                      Using polling engine: epoll1
    D0610 03:52:21.205954778      14 dns_resolver_ares.cc:822]             Using ares dns resolver
    D0610 03:52:21.206330022      14 lb_policy_registry.cc:46]             registering LB policy factory for "priority_experimental"
    D0610 03:52:21.206338673      14 lb_policy_registry.cc:46]             registering LB policy factory for "outlier_detection_experimental"
    D0610 03:52:21.206342037      14 lb_policy_registry.cc:46]             registering LB policy factory for "weighted_target_experimental"
    D0610 03:52:21.206344872      14 lb_policy_registry.cc:46]             registering LB policy factory for "pick_first"
    D0610 03:52:21.206347733      14 lb_policy_registry.cc:46]             registering LB policy factory for "round_robin"
    D0610 03:52:21.206350470      14 lb_policy_registry.cc:46]             registering LB policy factory for "weighted_round_robin_experimental"
    D0610 03:52:21.206356545      14 lb_policy_registry.cc:46]             registering LB policy factory for "ring_hash_experimental"
    D0610 03:52:21.206371806      14 lb_policy_registry.cc:46]             registering LB policy factory for "grpclb"
    D0610 03:52:21.206398664      14 lb_policy_registry.cc:46]             registering LB policy factory for "rls_experimental"
    D0610 03:52:21.206417122      14 lb_policy_registry.cc:46]             registering LB policy factory for "xds_cluster_manager_experimental"
    D0610 03:52:21.206424670      14 lb_policy_registry.cc:46]             registering LB policy factory for "xds_cluster_impl_experimental"
    D0610 03:52:21.206428545      14 lb_policy_registry.cc:46]             registering LB policy factory for "cds_experimental"
    D0610 03:52:21.206434468      14 lb_policy_registry.cc:46]             registering LB policy factory for "xds_cluster_resolver_experimental"
    D0610 03:52:21.206437680      14 lb_policy_registry.cc:46]             registering LB policy factory for "xds_override_host_experimental"
    D0610 03:52:21.206440617      14 lb_policy_registry.cc:46]             registering LB policy factory for "xds_wrr_locality_experimental"
    D0610 03:52:21.206444328      14 certificate_provider_registry.cc:35]  registering certificate provider factory for "file_watcher"
    I0610 03:52:21.208642985      14 socket_utils_common_posix.cc:408]     Disabling AF_INET6 sockets because ::1 is not available.
    I0610 03:52:21.228206861     253 socket_utils_common_posix.cc:337]     TCP_USER_TIMEOUT is available. TCP_USER_TIMEOUT will be used thereafter
    E0610 03:52:21.235728907     253 oauth2_credentials.cc:236]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2023-06-10T03:52:21.2357118+00:00", grpc_status:2}



```python
#데이터 불러오기
epochs=100
rootPath="../input/plant-pathology-2020-fgvc7/"
images= "images/"
test= "test.csv"
train = "train.csv"
result = "sample_submission.csv"

submission = pd.read_csv(rootPath+result)
testData = pd.read_csv(rootPath+test)
trainData = pd.read_csv(rootPath+train)
plusData=pd.read_csv("/kaggle/input/pluslabel/plus.csv")

```


```python
#데이터불러오는 함수
def originPath(name):
    return rootPath + 'images/' + name + '.jpg'
#추가 데이터불러오는 함수
def plusPath(name):
    return "../input/plusdata/" + name + '.jpg'
```


```python
#데이터불러오는코드
testPaths = testData.image_id.apply(originPath).values
trainPaths = trainData.image_id.apply(originPath).values
plusPaths = plusData.image_id.apply(plusPath).values
trainLabels = np.float32(trainData.loc[:, 'healthy':'scab'].values)
plusLabels = np.float32(plusData.loc[:, 'healthy':'scab'].values)
trainPaths, validPaths, trainLabels, validLabels =train_test_split(trainPaths, trainLabels, test_size=0.15)
```


```python
#원래데이터에 추가 데이터 합치기
trainLabels=np.concatenate((trainLabels,plusLabels))
```


```python
#원래데이터 경로에 추가경로 합치기
trainPaths=np.concatenate((trainPaths,plusPaths))
```


```python
#2048 1365 3 데이터의 shape
```


```python
1365//4#데이터가 큼으로 비율대로 자르기
```




    341




```python
trainLabels.shape
```




    (2546, 4)




```python
height=341
width=514
```


```python
#데이터 전처리 함수
def dataPreprocessing(filename, label=None, image_size=(width, height)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label
    
#데이터 증가 함수
def dataAugment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:    
        return image
    return image, label
```


```python
#TPU설정
AUTO = tf.data.experimental.AUTOTUNE
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
#TPU상태에따른 바치사이즈 조정
batchSize = 16 * strategy.num_replicas_in_sync
```

    INFO:tensorflow:Deallocate tpu buffers before initializing tpu system.
    INFO:tensorflow:Initializing the TPU system: local
    INFO:tensorflow:Finished initializing TPU system.


    WARNING:absl:`tf.distribute.experimental.TPUStrategy` is deprecated, please use  the non experimental symbol `tf.distribute.TPUStrategy` instead.


    INFO:tensorflow:Found TPU system:


    INFO:tensorflow:Found TPU system:


    INFO:tensorflow:*** Num TPU Cores: 8


    INFO:tensorflow:*** Num TPU Cores: 8


    INFO:tensorflow:*** Num TPU Workers: 1


    INFO:tensorflow:*** Num TPU Workers: 1


    INFO:tensorflow:*** Num TPU Cores Per Worker: 8


    INFO:tensorflow:*** Num TPU Cores Per Worker: 8


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:0, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:0, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:1, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:1, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:2, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:2, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:3, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:3, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:4, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:4, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:5, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:5, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:6, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:6, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:7, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU:7, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)



```python
#이미지 불러옴
trainDataset = (
    tf.data.Dataset
    .from_tensor_slices((trainPaths, trainLabels))
    .map(dataPreprocessing, num_parallel_calls=AUTO)
    .map(dataAugment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(batchSize)
    .prefetch(AUTO)
)

validDataset = (
    tf.data.Dataset
    .from_tensor_slices((validPaths, validLabels))
    .map(dataPreprocessing, num_parallel_calls=AUTO)
    .batch(batchSize)
    .cache()
    .prefetch(AUTO)
)

testDataset = (
    tf.data.Dataset
    .from_tensor_slices(testPaths)
    .map(dataPreprocessing, num_parallel_calls=AUTO)
    .batch(batchSize)
)

```


```python
from tensorflow.keras.models import clone_model
class Combination():
    
    def __init__(self):
        self.model=None
        self.optimizer=None
    
    def setModel(self,Model):
        model=Model
        self.model=model.getModel()
        print(self.model.summary())
    
    def setOptimizer(self,Optimizer):
        
        self.optimizer=Optimizer
        print(self.optimizer)
    
    
        
        
    
    def onLearning(self,epochs=30):
        
        with strategy.scope():
        


            
            self.model.compile(optimizer=self.optimizer,
                          loss = 'categorical_crossentropy',
                          metrics=['categorical_accuracy'])
        
        stepPerEpoch = trainLabels.shape[0] // batchSize
        history = self.model.fit(
                    trainDataset,
                    epochs=epochs,
                    steps_per_epoch=stepPerEpoch,
                    validation_data=validDataset)
        return history
```


```python
class Model:
    def __init__(self):
        pass
    def getModel():
        pass
```


```python
class Optimizer:
    def __init__(self):
        pass
    def getOptimizer():
        pass
```


```python
#모델 불러오기
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2
```


```python
class Vgg16(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([
                            L.BatchNormalization(input_shape=(width, height, 3)),
                            VGG16(input_shape=(width, height, 3),
                                                      weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax',kernel_initializer='he_normal')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Vgg19(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([VGG19(input_shape=(width, height, 3),
                                                    weights="imagenet",
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Inceptionv3(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([InceptionV3(
                                                    weights="imagenet",
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.3),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax',kernel_initializer='he_normal')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet50(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet50(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.2),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet101(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet101(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.2),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet152(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet152(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.2),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet50v2(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet50V2(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet101v2(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet101V2(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.5),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class Resnet152v2(Model):
    def __init__(self):
        with strategy.scope():    
            self.Model=tf.keras.Sequential([ResNet152V2(input_shape=(width, height, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                        
                                        L.GlobalAveragePooling2D(),
                                        L.Dropout(0.5),
                                        L.Dense(trainLabels.shape[1],
                                                activation='softmax')])
        
    def getModel(self):
        return self.Model
    
```


```python
class SGD(Optimizer):
    def setSchedules(self,schedules):    
        self.Optim=tf.keras.optimizers.SGD(momentum=0.9,learning_rate=schedules)
        return self.Optim
```


```python
class Adagrad(Optimizer):
    def setSchedules(self,schedules):    
        self.Optim=tf.keras.optimizers.Adagrad(epsilon=1e-6,learning_rate=schedules)
        return self.Optim
```


```python
class RMSprop(Optimizer):
    def setSchedules(self,schedules):    
        self.Optim=tf.keras.optimizers.RMSprop(rho=0.9, epsilon=1e-06,learning_rate=schedules)
        return self.Optim
    
        
```


```python
class Adam(Optimizer):
    def setSchedules(self,schedules):    
        self.Optim=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999,learning_rate=schedules)
        return self.Optim
```


```python
# models={"Vgg16":Vgg16(),"Vgg19":Vgg19(),"Inceptionv3":Inceptionv3()\
#         ,"Resnet50":Resnet50(),"Resnet101":Resnet101(),"Resnet152":Resnet152()\
#         ,"Resnet50v2":Resnet50v2(),"Resnet101v2":Resnet101v2(),"Resnet152v2":Resnet152v2()}
```


```python
# optimizers={"SGD":SGD(),"Adagrad":Adagrad(),"RMSprop":RMSprop(),"Adam":Adam()}
```


```python
# learningRateSchedulers={"CosineDecay":schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.0)\
#                        ,"CosineDecayRestarts":schedules.CosineDecayRestarts(initial_learning_rate=0.001, t_mul=2.0,m_mul=1.0,first_decay_steps=1000, alpha=0.001)\
#                        ,"ExponentialDecay":schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=50,decay_rate=0.96,staircase=True)\
#                        ,"InverseTimeDecay":schedules.InverseTimeDecay(initial_learning_rate = 0.01,decay_steps = 1.0,decay_rate = 0.5)
#                        }
```


```python
models={"Inceptionv3":Inceptionv3(),"Resnet101v2":Resnet101v2()}
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87910968/87910968 [==============================] - 1s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    171317808/171317808 [==============================] - 1s 0us/step



```python
optimizers={"SGD":SGD()}
```


```python
learningRateSchedulers={"CosineDecayRestarts":schedules.CosineDecayRestarts(initial_learning_rate=0.001, t_mul=2.0,m_mul=1.0,first_decay_steps=1000, alpha=0.001)}
```


```python
combination=Combination()
```


```python
def display(training, validation, title,yTitle,epochs):     
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(1, epochs+1), mode='lines+markers', y=training, marker=dict(color="#dc143c"),name="Train"))

        fig.add_trace(
            go.Scatter(x=np.arange(1, epochs+1), mode='lines+markers', y=validation, marker=dict(color="#0080ff"),
                   name="Validation"))

        fig.update_layout(title_text=title, yaxis_title=yTitle, xaxis_title="Epochs", template="plotly_dark")
        fig.show()
```


```python

def performanceCheck(epochs):
    for modelName,model in models.items():
#         weights=model.getModel().get_weights()
        for optimizerName,optimizer in optimizers.items():
            for learningRateSchedulerName,learningRateScheduler in learningRateSchedulers.items():
#                 model.getModel().set_weights(weights)
                combination.setModel(model)
                combination.setOptimizer(optimizer.setSchedules(learningRateScheduler))
                history=combination.onLearning(epochs=epochs)
                trainAcc=history.history['categorical_accuracy']
                evalAcc=history.history['val_categorical_accuracy']
                trainLoss=history.history['loss']
                evalLoss=history.history['val_loss']
                title=modelName+" "+optimizerName+" "+learningRateSchedulerName+" "
                display(trainAcc,evalAcc,title+"Accuracy","Accuracy",epochs)
                display(trainLoss,evalLoss,title+"Loss","Loss",epochs)
```


```python
performanceCheck(30)
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     inception_v3 (Functional)   (None, None, None, 2048)  21802784  
                                                                     
     global_average_pooling2d (G  (None, 2048)             0         
     lobalAveragePooling2D)                                          
                                                                     
     dropout (Dropout)           (None, 2048)              0         
                                                                     
     dense (Dense)               (None, 4)                 8196      
                                                                     
    =================================================================
    Total params: 21,810,980
    Trainable params: 21,776,548
    Non-trainable params: 34,432
    _________________________________________________________________
    None
    <keras.optimizers.sgd.SGD object at 0x7ef48c3a5430>
    Epoch 1/30


    2023-06-10 03:54:09.216458: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.
    2023-06-10 03:54:09.830044: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.


    19/19 [==============================] - ETA: 0s - loss: 1.3575 - categorical_accuracy: 0.3516

    2023-06-10 03:55:23.514984: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.
    2023-06-10 03:55:23.832619: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.


    19/19 [==============================] - 135s 3s/step - loss: 1.3575 - categorical_accuracy: 0.3516 - val_loss: 1.1679 - val_categorical_accuracy: 0.4745
    Epoch 2/30
    19/19 [==============================] - 10s 531ms/step - loss: 1.2332 - categorical_accuracy: 0.4823 - val_loss: 0.9829 - val_categorical_accuracy: 0.7336
    Epoch 3/30
    19/19 [==============================] - 9s 491ms/step - loss: 1.0345 - categorical_accuracy: 0.6188 - val_loss: 0.6806 - val_categorical_accuracy: 0.8431
    Epoch 4/30
    19/19 [==============================] - 11s 593ms/step - loss: 0.8397 - categorical_accuracy: 0.7052 - val_loss: 0.5487 - val_categorical_accuracy: 0.8431
    Epoch 5/30
    19/19 [==============================] - 10s 526ms/step - loss: 0.6914 - categorical_accuracy: 0.7496 - val_loss: 0.4939 - val_categorical_accuracy: 0.8686
    Epoch 6/30
    19/19 [==============================] - 9s 484ms/step - loss: 0.6822 - categorical_accuracy: 0.7492 - val_loss: 0.4544 - val_categorical_accuracy: 0.8796
    Epoch 7/30
    19/19 [==============================] - 9s 489ms/step - loss: 0.6024 - categorical_accuracy: 0.7775 - val_loss: 0.3885 - val_categorical_accuracy: 0.8832
    Epoch 8/30
    19/19 [==============================] - 9s 503ms/step - loss: 0.5473 - categorical_accuracy: 0.8022 - val_loss: 0.3391 - val_categorical_accuracy: 0.8869
    Epoch 9/30
    19/19 [==============================] - 9s 506ms/step - loss: 0.5514 - categorical_accuracy: 0.8039 - val_loss: 0.3074 - val_categorical_accuracy: 0.9161
    Epoch 10/30
    19/19 [==============================] - 9s 497ms/step - loss: 0.4885 - categorical_accuracy: 0.8326 - val_loss: 0.2559 - val_categorical_accuracy: 0.9307
    Epoch 11/30
    19/19 [==============================] - 11s 604ms/step - loss: 0.4491 - categorical_accuracy: 0.8409 - val_loss: 0.2386 - val_categorical_accuracy: 0.9270
    Epoch 12/30
    19/19 [==============================] - 10s 550ms/step - loss: 0.4155 - categorical_accuracy: 0.8639 - val_loss: 0.2153 - val_categorical_accuracy: 0.9270
    Epoch 13/30
    19/19 [==============================] - 9s 475ms/step - loss: 0.4389 - categorical_accuracy: 0.8499 - val_loss: 0.1818 - val_categorical_accuracy: 0.9416
    Epoch 14/30
    19/19 [==============================] - 9s 489ms/step - loss: 0.4001 - categorical_accuracy: 0.8639 - val_loss: 0.1773 - val_categorical_accuracy: 0.9562
    Epoch 15/30
    19/19 [==============================] - 9s 499ms/step - loss: 0.3650 - categorical_accuracy: 0.8734 - val_loss: 0.1597 - val_categorical_accuracy: 0.9526
    Epoch 16/30
    19/19 [==============================] - 9s 483ms/step - loss: 0.3541 - categorical_accuracy: 0.8820 - val_loss: 0.1513 - val_categorical_accuracy: 0.9526
    Epoch 17/30
    19/19 [==============================] - 10s 552ms/step - loss: 0.3314 - categorical_accuracy: 0.8869 - val_loss: 0.1357 - val_categorical_accuracy: 0.9599
    Epoch 18/30
    19/19 [==============================] - 9s 494ms/step - loss: 0.3425 - categorical_accuracy: 0.8890 - val_loss: 0.1397 - val_categorical_accuracy: 0.9562
    Epoch 19/30
    19/19 [==============================] - 9s 488ms/step - loss: 0.3106 - categorical_accuracy: 0.9017 - val_loss: 0.1340 - val_categorical_accuracy: 0.9599
    Epoch 20/30
    19/19 [==============================] - 9s 504ms/step - loss: 0.3187 - categorical_accuracy: 0.8980 - val_loss: 0.1333 - val_categorical_accuracy: 0.9635
    Epoch 21/30
    19/19 [==============================] - 10s 507ms/step - loss: 0.2824 - categorical_accuracy: 0.9025 - val_loss: 0.1212 - val_categorical_accuracy: 0.9708
    Epoch 22/30
    19/19 [==============================] - 9s 489ms/step - loss: 0.2926 - categorical_accuracy: 0.9030 - val_loss: 0.1139 - val_categorical_accuracy: 0.9745
    Epoch 23/30
    19/19 [==============================] - 10s 542ms/step - loss: 0.2565 - categorical_accuracy: 0.9145 - val_loss: 0.1105 - val_categorical_accuracy: 0.9672
    Epoch 24/30
    19/19 [==============================] - 9s 504ms/step - loss: 0.2572 - categorical_accuracy: 0.9165 - val_loss: 0.1083 - val_categorical_accuracy: 0.9745
    Epoch 25/30
    19/19 [==============================] - 9s 502ms/step - loss: 0.2441 - categorical_accuracy: 0.9198 - val_loss: 0.1046 - val_categorical_accuracy: 0.9708
    Epoch 26/30
    19/19 [==============================] - 9s 505ms/step - loss: 0.2464 - categorical_accuracy: 0.9174 - val_loss: 0.1006 - val_categorical_accuracy: 0.9708
    Epoch 27/30
    19/19 [==============================] - 9s 493ms/step - loss: 0.2145 - categorical_accuracy: 0.9289 - val_loss: 0.1022 - val_categorical_accuracy: 0.9745
    Epoch 28/30
    19/19 [==============================] - 10s 554ms/step - loss: 0.2301 - categorical_accuracy: 0.9248 - val_loss: 0.0991 - val_categorical_accuracy: 0.9708
    Epoch 29/30
    19/19 [==============================] - 10s 555ms/step - loss: 0.2174 - categorical_accuracy: 0.9309 - val_loss: 0.0985 - val_categorical_accuracy: 0.9745
    Epoch 30/30
    19/19 [==============================] - 10s 501ms/step - loss: 0.2251 - categorical_accuracy: 0.9235 - val_loss: 0.0990 - val_categorical_accuracy: 0.9745



<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.24.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="6fe0d990-7863-487c-b010-35a325054296" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6fe0d990-7863-487c-b010-35a325054296")) {                    Plotly.newPlot(                        "6fe0d990-7863-487c-b010-35a325054296",                        [{"marker":{"color":"#dc143c"},"mode":"lines+markers","name":"Train","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[0.3515625,0.48231908679008484,0.6188322305679321,0.7051809430122375,0.7495887875556946,0.7491776347160339,0.7775493264198303,0.8022204041481018,0.8038651347160339,0.8326480388641357,0.8408716917037964,0.8638980388641357,0.8499177694320679,0.8638980388641357,0.8733552694320679,0.8819901347160339,0.8869243264198303,0.8889802694320679,0.9017269611358643,0.8980262875556946,0.9025493264198303,0.9029605388641357,0.9144737124443054,0.9165295958518982,0.9198190569877625,0.9173519611358643,0.9288651347160339,0.9247533082962036,0.9309210777282715,0.9235197305679321],"type":"scatter"},{"marker":{"color":"#0080ff"},"mode":"lines+markers","name":"Validation","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[0.47445255517959595,0.7335766553878784,0.8430656790733337,0.8430656790733337,0.8686131238937378,0.8795620203018188,0.8832116723060608,0.8868613243103027,0.9160584211349487,0.930656909942627,0.9270073175430298,0.9270073175430298,0.9416058659553528,0.956204354763031,0.9525547623634338,0.9525547623634338,0.959854006767273,0.956204354763031,0.959854006767273,0.9635036587715149,0.970802903175354,0.974452555179596,0.9671533107757568,0.974452555179596,0.970802903175354,0.970802903175354,0.974452555179596,0.970802903175354,0.974452555179596,0.974452555179596],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"geo":{"bgcolor":"rgb(17,17,17)","lakecolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","showlakes":true,"showland":true,"subunitcolor":"#506784"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"dark"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"sliderdefaults":{"bgcolor":"#C8D4E3","bordercolor":"rgb(17,17,17)","borderwidth":1,"tickwidth":0},"ternary":{"aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"xaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2}}},"title":{"text":"Inceptionv3 SGD CosineDecayRestarts Accuracy"},"yaxis":{"title":{"text":"Accuracy"}},"xaxis":{"title":{"text":"Epochs"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('6fe0d990-7863-487c-b010-35a325054296');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



<div>                            <div id="a96af8b6-32e8-45c5-bc35-15be97fde604" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a96af8b6-32e8-45c5-bc35-15be97fde604")) {                    Plotly.newPlot(                        "a96af8b6-32e8-45c5-bc35-15be97fde604",                        [{"marker":{"color":"#dc143c"},"mode":"lines+markers","name":"Train","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[1.3575106859207153,1.2331663370132446,1.0345261096954346,0.8397260904312134,0.6913638114929199,0.6822370886802673,0.6023992300033569,0.547330379486084,0.5513661503791809,0.48845401406288147,0.4490939676761627,0.41545918583869934,0.4388968348503113,0.40010377764701843,0.3650321066379547,0.35414913296699524,0.33142611384391785,0.3425161838531494,0.31060442328453064,0.318727046251297,0.2824431359767914,0.2925732135772705,0.2565165162086487,0.25715628266334534,0.24409067630767822,0.24641844630241394,0.2145344316959381,0.23012498021125793,0.2174161672592163,0.22507838904857635],"type":"scatter"},{"marker":{"color":"#0080ff"},"mode":"lines+markers","name":"Validation","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[1.1679253578186035,0.9828733801841736,0.6806005835533142,0.5487240552902222,0.4938836693763733,0.4544224143028259,0.38849908113479614,0.3391016125679016,0.30741119384765625,0.25587499141693115,0.2385586053133011,0.21531255543231964,0.18177440762519836,0.17733722925186157,0.15967945754528046,0.1513497680425644,0.1356547474861145,0.13971903920173645,0.13396841287612915,0.13329678773880005,0.12119048088788986,0.11393376439809799,0.11045558005571365,0.1083131954073906,0.10455073416233063,0.1006166860461235,0.10219328850507736,0.09908106178045273,0.09853266924619675,0.09895877540111542],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"geo":{"bgcolor":"rgb(17,17,17)","lakecolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","showlakes":true,"showland":true,"subunitcolor":"#506784"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"dark"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"sliderdefaults":{"bgcolor":"#C8D4E3","bordercolor":"rgb(17,17,17)","borderwidth":1,"tickwidth":0},"ternary":{"aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"xaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2}}},"title":{"text":"Inceptionv3 SGD CosineDecayRestarts Loss"},"yaxis":{"title":{"text":"Loss"}},"xaxis":{"title":{"text":"Epochs"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a96af8b6-32e8-45c5-bc35-15be97fde604');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     resnet101v2 (Functional)    (None, 17, 11, 2048)      42626560  
                                                                     
     global_average_pooling2d_1   (None, 2048)             0         
     (GlobalAveragePooling2D)                                        
                                                                     
     dropout_1 (Dropout)         (None, 2048)              0         
                                                                     
     dense_1 (Dense)             (None, 4)                 8196      
                                                                     
    =================================================================
    Total params: 42,634,756
    Trainable params: 42,537,092
    Non-trainable params: 97,664
    _________________________________________________________________
    None
    <keras.optimizers.sgd.SGD object at 0x7ef48c2246d0>
    Epoch 1/30


    2023-06-10 04:01:15.757060: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.
    2023-06-10 04:01:16.640682: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.


    19/19 [==============================] - ETA: 0s - loss: 1.5281 - categorical_accuracy: 0.3104

    2023-06-10 04:02:16.986371: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.
    2023-06-10 04:02:17.383824: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp.


    19/19 [==============================] - 112s 2s/step - loss: 1.5281 - categorical_accuracy: 0.3104 - val_loss: 1.2059 - val_categorical_accuracy: 0.4891
    Epoch 2/30
    19/19 [==============================] - 10s 548ms/step - loss: 1.2129 - categorical_accuracy: 0.4856 - val_loss: 0.9957 - val_categorical_accuracy: 0.7226
    Epoch 3/30
    19/19 [==============================] - 10s 518ms/step - loss: 0.9517 - categorical_accuracy: 0.6369 - val_loss: 0.6501 - val_categorical_accuracy: 0.7920
    Epoch 4/30
    19/19 [==============================] - 10s 509ms/step - loss: 0.7471 - categorical_accuracy: 0.7208 - val_loss: 0.5837 - val_categorical_accuracy: 0.8212
    Epoch 5/30
    19/19 [==============================] - 11s 575ms/step - loss: 0.6520 - categorical_accuracy: 0.7574 - val_loss: 0.5254 - val_categorical_accuracy: 0.8394
    Epoch 6/30
    19/19 [==============================] - 10s 518ms/step - loss: 0.5773 - categorical_accuracy: 0.7891 - val_loss: 0.4531 - val_categorical_accuracy: 0.8759
    Epoch 7/30
    19/19 [==============================] - 10s 522ms/step - loss: 0.5270 - categorical_accuracy: 0.8018 - val_loss: 0.3937 - val_categorical_accuracy: 0.8942
    Epoch 8/30
    19/19 [==============================] - 11s 562ms/step - loss: 0.4898 - categorical_accuracy: 0.8195 - val_loss: 0.3466 - val_categorical_accuracy: 0.9051
    Epoch 9/30
    19/19 [==============================] - 10s 529ms/step - loss: 0.4375 - categorical_accuracy: 0.8475 - val_loss: 0.3108 - val_categorical_accuracy: 0.9015
    Epoch 10/30
    19/19 [==============================] - 10s 516ms/step - loss: 0.4078 - categorical_accuracy: 0.8528 - val_loss: 0.2389 - val_categorical_accuracy: 0.9088
    Epoch 11/30
    19/19 [==============================] - 10s 523ms/step - loss: 0.3784 - categorical_accuracy: 0.8676 - val_loss: 0.2476 - val_categorical_accuracy: 0.9051
    Epoch 12/30
    19/19 [==============================] - 10s 514ms/step - loss: 0.3302 - categorical_accuracy: 0.8808 - val_loss: 0.2020 - val_categorical_accuracy: 0.9234
    Epoch 13/30
    19/19 [==============================] - 10s 530ms/step - loss: 0.3157 - categorical_accuracy: 0.8931 - val_loss: 0.2107 - val_categorical_accuracy: 0.9234
    Epoch 14/30
    19/19 [==============================] - 11s 604ms/step - loss: 0.3035 - categorical_accuracy: 0.8943 - val_loss: 0.1565 - val_categorical_accuracy: 0.9380
    Epoch 15/30
    19/19 [==============================] - 10s 523ms/step - loss: 0.2637 - categorical_accuracy: 0.9190 - val_loss: 0.1668 - val_categorical_accuracy: 0.9416
    Epoch 16/30
    19/19 [==============================] - 10s 528ms/step - loss: 0.2557 - categorical_accuracy: 0.9153 - val_loss: 0.1502 - val_categorical_accuracy: 0.9380
    Epoch 17/30
    19/19 [==============================] - 10s 508ms/step - loss: 0.2498 - categorical_accuracy: 0.9132 - val_loss: 0.1393 - val_categorical_accuracy: 0.9599
    Epoch 18/30
    19/19 [==============================] - 10s 515ms/step - loss: 0.2187 - categorical_accuracy: 0.9289 - val_loss: 0.1293 - val_categorical_accuracy: 0.9562
    Epoch 19/30
    19/19 [==============================] - 10s 533ms/step - loss: 0.2051 - categorical_accuracy: 0.9391 - val_loss: 0.1276 - val_categorical_accuracy: 0.9489
    Epoch 20/30
    19/19 [==============================] - 10s 506ms/step - loss: 0.2013 - categorical_accuracy: 0.9301 - val_loss: 0.1173 - val_categorical_accuracy: 0.9672
    Epoch 21/30
    19/19 [==============================] - 10s 524ms/step - loss: 0.1747 - categorical_accuracy: 0.9478 - val_loss: 0.1122 - val_categorical_accuracy: 0.9599
    Epoch 22/30
    19/19 [==============================] - 10s 519ms/step - loss: 0.1851 - categorical_accuracy: 0.9371 - val_loss: 0.1092 - val_categorical_accuracy: 0.9599
    Epoch 23/30
    19/19 [==============================] - 10s 526ms/step - loss: 0.1703 - categorical_accuracy: 0.9437 - val_loss: 0.1115 - val_categorical_accuracy: 0.9745
    Epoch 24/30
    19/19 [==============================] - 10s 524ms/step - loss: 0.1656 - categorical_accuracy: 0.9527 - val_loss: 0.1080 - val_categorical_accuracy: 0.9599
    Epoch 25/30
    19/19 [==============================] - 10s 535ms/step - loss: 0.1552 - categorical_accuracy: 0.9531 - val_loss: 0.1064 - val_categorical_accuracy: 0.9562
    Epoch 26/30
    19/19 [==============================] - 10s 545ms/step - loss: 0.1366 - categorical_accuracy: 0.9552 - val_loss: 0.1066 - val_categorical_accuracy: 0.9635
    Epoch 27/30
    19/19 [==============================] - 11s 604ms/step - loss: 0.1345 - categorical_accuracy: 0.9556 - val_loss: 0.1021 - val_categorical_accuracy: 0.9562
    Epoch 28/30
    19/19 [==============================] - 10s 523ms/step - loss: 0.1455 - categorical_accuracy: 0.9519 - val_loss: 0.1140 - val_categorical_accuracy: 0.9562
    Epoch 29/30
    19/19 [==============================] - 10s 532ms/step - loss: 0.1216 - categorical_accuracy: 0.9622 - val_loss: 0.1054 - val_categorical_accuracy: 0.9599
    Epoch 30/30
    19/19 [==============================] - 10s 522ms/step - loss: 0.1112 - categorical_accuracy: 0.9683 - val_loss: 0.1004 - val_categorical_accuracy: 0.9708



<div>                            <div id="a8e07266-f9ea-49cb-92fa-b0e18a238e6e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a8e07266-f9ea-49cb-92fa-b0e18a238e6e")) {                    Plotly.newPlot(                        "a8e07266-f9ea-49cb-92fa-b0e18a238e6e",                        [{"marker":{"color":"#dc143c"},"mode":"lines+markers","name":"Train","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[0.31044408679008484,0.4856085479259491,0.6369243264198303,0.7208059430122375,0.7574012875556946,0.7890625,0.8018091917037964,0.8194901347160339,0.8474506735801697,0.8527960777282715,0.8675987124443054,0.8807565569877625,0.8930920958518982,0.8943256735801697,0.9189966917037964,0.9152960777282715,0.9132401347160339,0.9288651347160339,0.9391447305679321,0.9300987124443054,0.9477795958518982,0.9370887875556946,0.9436677694320679,0.9527137875556946,0.953125,0.9551809430122375,0.9555920958518982,0.9518914222717285,0.9621710777282715,0.9683387875556946],"type":"scatter"},{"marker":{"color":"#0080ff"},"mode":"lines+markers","name":"Validation","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[0.48905110359191895,0.7226277589797974,0.7919707894325256,0.8211678862571716,0.8394160866737366,0.8759124279022217,0.8941605687141418,0.9051094651222229,0.9014598727226257,0.9087591171264648,0.9051094651222229,0.9233576655387878,0.9233576655387878,0.9379562139511108,0.9416058659553528,0.9379562139511108,0.959854006767273,0.956204354763031,0.9489051103591919,0.9671533107757568,0.959854006767273,0.959854006767273,0.974452555179596,0.959854006767273,0.956204354763031,0.9635036587715149,0.956204354763031,0.956204354763031,0.959854006767273,0.970802903175354],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"geo":{"bgcolor":"rgb(17,17,17)","lakecolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","showlakes":true,"showland":true,"subunitcolor":"#506784"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"dark"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"sliderdefaults":{"bgcolor":"#C8D4E3","bordercolor":"rgb(17,17,17)","borderwidth":1,"tickwidth":0},"ternary":{"aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"xaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2}}},"title":{"text":"Resnet101v2 SGD CosineDecayRestarts Accuracy"},"yaxis":{"title":{"text":"Accuracy"}},"xaxis":{"title":{"text":"Epochs"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a8e07266-f9ea-49cb-92fa-b0e18a238e6e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



<div>                            <div id="34d6882b-ff6b-411c-9cc8-7a20958b8655" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("34d6882b-ff6b-411c-9cc8-7a20958b8655")) {                    Plotly.newPlot(                        "34d6882b-ff6b-411c-9cc8-7a20958b8655",                        [{"marker":{"color":"#dc143c"},"mode":"lines+markers","name":"Train","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[1.5280652046203613,1.2129074335098267,0.9516862630844116,0.7470691800117493,0.651993453502655,0.5773174166679382,0.5270390510559082,0.4898415207862854,0.437452107667923,0.4078429937362671,0.3783981204032898,0.330175518989563,0.315677672624588,0.30352556705474854,0.2636817991733551,0.2556808292865753,0.24983453750610352,0.21871855854988098,0.20512418448925018,0.20132355391979218,0.17470921576023102,0.18508148193359375,0.17033424973487854,0.16564537584781647,0.1551521271467209,0.13658924400806427,0.13446584343910217,0.14554090797901154,0.12157148867845535,0.11119449883699417],"type":"scatter"},{"marker":{"color":"#0080ff"},"mode":"lines+markers","name":"Validation","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"y":[1.2058583498001099,0.9956871271133423,0.6501286029815674,0.5837215185165405,0.5253880023956299,0.45312467217445374,0.3936689794063568,0.346596360206604,0.3108144700527191,0.23894824087619781,0.247593954205513,0.2020389884710312,0.21069413423538208,0.1564655750989914,0.1668286770582199,0.15022125840187073,0.13932716846466064,0.1293375939130783,0.12755922973155975,0.11726746708154678,0.11218534409999847,0.10919755697250366,0.11145524680614471,0.10799943655729294,0.1063525527715683,0.10660985112190247,0.10206755995750427,0.1139799952507019,0.10544601827859879,0.10044953972101212],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"geo":{"bgcolor":"rgb(17,17,17)","lakecolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","showlakes":true,"showland":true,"subunitcolor":"#506784"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"dark"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"sliderdefaults":{"bgcolor":"#C8D4E3","bordercolor":"rgb(17,17,17)","borderwidth":1,"tickwidth":0},"ternary":{"aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"xaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2}}},"title":{"text":"Resnet101v2 SGD CosineDecayRestarts Loss"},"yaxis":{"title":{"text":"Loss"}},"xaxis":{"title":{"text":"Epochs"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('34d6882b-ff6b-411c-9cc8-7a20958b8655');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
model=models['Resnet101v2'].getModel()
```


```python
testModel=clone_model(model)
```


```python
y=testModel.predict(testDataset)
```

    15/15 [==============================] - 139s 9s/step



```python
submission=pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>healthy</th>
      <th>multiple_diseases</th>
      <th>rust</th>
      <th>scab</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Test_0</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Test_1</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test_2</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Test_3</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Test_4</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1816</th>
      <td>Test_1816</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1817</th>
      <td>Test_1817</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1818</th>
      <td>Test_1818</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1819</th>
      <td>Test_1819</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1820</th>
      <td>Test_1820</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
<p>1821 rows × 5 columns</p>
</div>




```python
result=pd.DataFrame({
    "image_id":submission['image_id'],
    "healthy":y[:,0],
    "multiple_diseases":y[:,1],
    "rust":y[:,2],
    "scab":y[:,3]
})
```


```python
result.to_csv("submission.csv",index=False)
```


```python

```
