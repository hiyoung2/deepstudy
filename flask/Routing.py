# Routing

##########################################################################
# @app.route('/test')
# def ...
# 그냥 uri만 있는 상태

##########################################################################
# @app.route('/test', methods=[ 'POST', 'PUT'])
# def ...
# method를 추가, 'POST', 'PUT'일 때만 아래 함수를 실행하라
# method를 따로 지정하지 않으면 default인 'GET'이 적용된다


##########################################################################
# @app.route('/test/<tid>')
# def test3(tid):
#     print("tid is", tid)
# parameter를 받는 것
# tid는 하나의 변수
# 예를 들어 'http:~~/boards/100' 이면 100번째 페이지의 게시글을 보여줌
# board 뒤에 1이든 2든 올 수 있다
# 변수는 여러 개 올 수가 있다, 뭐든 올 수 있다
# 예를 들어 '~~/boards/100/comments/?page=3
# boards 게시판의 100번째 글 중 comments 댓글 페이지 3번째 uri 
# 참고로 comments == param, ?page=3 == query param


##########################################################################
# @app.route('/test', defaults={'page': 'index'}) (1)
# @app.route('/test/<page>') (2)
# def xxx(page)

# (2)를 통해 page를 받을 건데, def 에서 page를 지정해줌
# if page값이 안 들어 왔을 때, (1)에서 설정한 index를 page값으로 주겠다는 의미

##########################################################################
# @app.route('/test', host='abc.com') # 도메인 부여
# @app.route('/test', redirect_to='/new_test')

# redirect_to : a.html이라고 쳤는데 b.html이 출력되는 것
# 참고로 foward는 주소가 안 바뀐 상태로 넘어가는 것


##########################################################################

# Rounting(Cont'd) : subdomaiin
# app.config['SERVER_NAME'] = 'local.com:5000'

# @app.route("/")
# def helloworld_local():
#     return "Hello Local.com!"

# @app.route("/", subdomain="g")
# def helloworld():
#     return "Hello G.Local.com!!!"

# naver.com -> 네이버 메인화면
# blog.naver.com -> 네이버의 블로그 메인화면
# 여기서 blog : subdomain
# 사실, naver.com에는 www. 이 생략된 상태
# 위 예제는 g/locl~~로 접속하면 Hello G.Local.com!!!이 출력되게 만든 것이다
# g == subdomain
# cafe면 subdomain="cafe" 등 이렇게 지정해서 페이지들을 분리할 수 있다



##########################################################################
# Request Parameter
# 사용예시는 pyweb 폴더 참고

# MultiDict Type
# ...get('<param name>', <default-value>, <type>)
#             (1)               (2)         (3)
# methods: get, getlist, clear, etc

# parameter 정말 많이 쓴다
# 대부분 파라미터 가지고 작업을 하기 때문에 잘 알아야 한다
# 파라미터의 dtype = MultiDict
# 딕셔너리 같은 건데 더 확장된 버전이다
# get 뒤에 (1) : 파라미터의 이름, (2) : 디폴트 값(파라미터 값이 정의 되지 않았을 때), (3) :int라고 하면 int형으로 받는다
# client로부터 server로 오는 것은 무엇이든 string 형태로 받기 때문에, 필요에 따라 type을 지정해 주면 된다

# GET
# request.args.get('q')

# POST
# request.form.get('p', 123)
# post를 쓸 때에는 args가 아닌 form을 써야 한다
# form submit 할 때는 POST로 보낸다
# 로그인 페이지라면 아이디와 패스워드를 form.get('id') or form.get('password')를 할 수 있다
# 오른쪽 123은 값을 안 줬을 때 default값으로 적용할 값


# GET or POST
# request.values.get('v')
# get, post가 헷갈린다면 values를 쓰면 된다
# get이든 post든 파라미터 모두 받는다

# Parameters
# request.args.getlist('qs')
# list형태로, 여러 가지 변수들을 한 번에 받을 수 있다 -> pyweb 참고


##########################################################################
# Request Parameter Custom Function Type
# request 처리용 함수

# from datetime import datetime, data
# def ymd(fmt):
#     def trans(date_str):
#         return datetime.strptime(date_str, fmt)
#     return trans

# @app.route('/dt')
# def dt():
#     datestr = request.values.get('date', date.tody(), type=ymd('%Y-%m-%d'))
#                              # parametername # default # type(현재 함수가 붙었음 , ymd 함수는 위에 정의되어 있음)
#     return "우리나라 시간 형식: " + str(datestr)