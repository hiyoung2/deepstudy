# 내가 만든 model을 저장해보자

'''
1. 모델 저장
모델을 저장하는 방법에는 2가지가 있다

1) 모델 구성 다음에 바로 저장하기
# 2의 모델 구성 하단에 
model.save('./위치/위치/.../파일명.h5) 라고 적어주면
입력한 위치에 파일이 생성된다
이 때는 딱! 모델 자체만 저장이 되고, 재사용시 load 해 주면 되는데
재사용 할 때 레이어를 추가해 줄 수 있다, 아웃풋 맞춰주는 것 주의!
모델만 저장했기 때문에 #2의 과정은 생략 될 수 있지만
compile , fit은 또 따로 해 줘야 한다

2) fit 과정 다음에 저장하기
이렇게 저장하면 모델 뿐만 아니라 fit 과정을 통해 나온 weight 값도 함께 저장이 된다
이렇게 저장한 모델을 재사용하려고 load를 하면 #2의 모델 구성 생략이 가능하고
compile, fit 과정도 함께 생략해서 실행 가능하다
바로 결과를 볼 수 있다

2. weight, 가중치 저장
fit 과정 다음에 (fit을 거쳐야 weight 값이 나오니까)
model.save_weights('./model/위치/.../파일명.h5') 라고 적어주면
weight 가 저장된다
재사용하기 위해 load를 한다면,
# 2 모델 구성, compile 과정을 입력해야 한다
딱 weight 만 저장된 것이라서 모델 구성이 필요한데, 기존 모델과 같은 구성이어야 쓸 수 있다!
이렇게 보면 weight만 저장하는 것은 쓸 일이 없어 보인다
선생님 말씀으로는 이것도 많이 쓰인다고한다
모델, weight를 함께 모두 저장해서 같이 불러오는 게 무조건적으로 좋을 순 없다고,,
그리고 weight 저장한 걸 불러와서 # 3 과정을 싹 다 주석처리 하면 돌아가지 않음
compile 과정도 입력해줘야 한다
fit 없이는 돌아간다

3. checkpoint 저장
callbacks에 들어 있는 함수 ModelCheckpoint를 사용했을 시에 monitor로 사용한 loss(또는 val_loss 등등 지정하기 나름)가
best일 때를 저장 할 수 있는데, 이 때는 model, weight도 모두 함께 저장된다!!
save_best_only = True, save_weights_only = False 로 지정해줘야한다
checkpoinit 를 저장해서 재사용하기 위해 load 하면
from keras.models import load_model
model = load_model('./위치/.../위치/파일명/hdf5) (cehckpoint는 hdf5 파일 형태로 저장된다)
를 입력하면 불러진다
그리고 #2, #3의 과정을 모두 주석처리 하여 실행 가능하다


# 각 저장된 것들 불러오기
1. 저장된 모델 load
from keras.models import load_model
model = load_model('./위치/파일명')
# 모델이 저장되어 있으므로 from keras.models import load_model 


2. 저장된 weight load
model.load_weights('./위치/파일명')
# 모델은 저장 안 되어있으니까 그냥 이렇게만 적어준다

3. 저장된 checkpoint load
from keras.models import laod_model
model = load_model('./위치/파일명')
# 모델이 저장되어 있으므로 from keras.models import load_model

# 요약
                      model      weight      loc
model save            O            O       모델구성다음 or fit 다음

weight save           X            O       fit 다음

cehckpoint save       O            O       fit 과정에서


저장하는 것들을 어떻게 써먹을까?
input, output에 주의해서
내가 짠 모델 사이에 우승자의 모델을 집어 넣을 수 있다?
튜닝은 필요하지만,,

'''