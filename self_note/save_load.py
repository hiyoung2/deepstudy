# model.save('/파일 저장할 위치(폴더명)/저장할 파일명.h5') : 모델 저장
# 1) compile, fit 과정 전, 모델 구성 다음에 저장하면 모델 자체만 저장된다
# 2) fit 과정 다음에 model.save 하면 내가 짠 모델과 가중치(weight) 모두 저장된다
# 2번처럼 저장한다면
# from keras.models import load_model
# model = load_model('./파일 저장할 위치(폴더명)/저장할 파일명.h5') 로 불러오면
# compile, fit 과정 없이 실행이 된다

# model.save_weights('/위치/파일명.h5')로 weight만 저장하면(fit다음에 해야함)
# load 했을 때, 모델과 compile 과정을 살려두면 실행된다(fit 과정은 없어도 된다)

# checkpoint 저장
# modelpath로 cehckpoint 저장할 위치를 만들어주고
# checkpoint = ModelCheckpoint(filepath, monitor, save_best_only, save_weights_only, verbose)로ㅓ
# checkpoint를 만들어주고 fit과정에서 callbacks로 호출해준다
# model.save(~~)로 checkpoint를 저장하면

