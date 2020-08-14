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

# MultiDict Type
# ...get('<param name>', <default-value>, <type>)
# methods: get, getlist, clear, etc

# GET
# request.args.get('q')

# POST
# request.from.get('p', 123)

# GET or POST
# request.values.get('v')

# Parameters
# request.args.getlist('qs')