# 오픈 도메인 사용자 경험 기반 대화 모델 V.1 (SW)

## 모델 설명
- 일상 대화와 QA, Tool calling 기능을 갖춘 **오픈 도메인 사용자 경험 기반 대화 모델 V.1 (SW)

## 모델 빌드 방법 및 사용법

1. 기본 docker image를 빌드한다.

```bash
cd base_docker/v0.2.90
docker build -t keti/llama_cpp-cuda12.1:v0.2.90 .
```

2. 모델 파일을 다운로드 받아 `docker/app/model` 아래로 복사한다.

| model name | URL |
| --- | --- |
| v0.59 | [Link](https://drive.google.com/file/d/1JPqgKAiMRapJpaV3exAEZPRWW757YvwY/view?usp=drive_link) |
| v0.58 | [Link](https://drive.google.com/file/d/1CUC1orqFIMxAwTIuiOnoQWLkFRbYkjEx/view?usp=sharing) |
| v0.56 | [Link](https://drive.google.com/file/d/1IGGzWhWjlZWKCBgwJ0yry_Wjwfr7CEgk/view?usp=sharing) |
| v0.51 | [Link](https://drive.google.com/file/d/1zaD491PwfujjjIrct_UDioL9xldEKg33/view?usp=sharing) |
| v0.43 | [Link](https://drive.google.com/file/d/1diEg8zuV-j_FUMaI2FxEYsvIV717FmrI/view?usp=drive_link) |

3. 어플리케이션 docker image를 빌드한다.

```bash
cd docker
docker build -t keti/mmc2_model:v0.58 .
```

4. 빌드된 docker image로 container를 실행한다.
```bash
docker run --gpus=all -p 5000:5000 --rm -it --name mmc2_model keti/mmc2_model:v0.58
```

5. `test.py` 파일을 이용하여 모델이 정상 동작하는지 확인한다.

```bash
cd docker
python test.py
```

- Resonse
```
200
<PreparedRequest [POST]>
{'choices': [{'finish_reason': 'tool_calls',
              'index': 0,
              'logprobs': None,
              'message': {'content': None,
                          'role': 'assistant',
                          'tool_calls': [{'function': {'arguments': '{"location": '
                                                                    '"New '
                                                                    'York, '
                                                                    'NY"}',
                                                       'name': 'get_rain_probability'},
                                          'id': 'call__0_get_rain_probability_655ab1ad-1e8b-42a3-b112-e7190208dcba',
                                          'type': 'function'},
                                         {'function': {'arguments': '{"location": '
                                                                    '"Seoul, '
                                                                    'South '
                                                                    'Korea"}',
                                                       'name': 'get_rain_probability'},
                                          'id': 'call__1_get_rain_probability_10eb60df-9b0d-425d-aa6b-9644f46474d3',
                                          'type': 'function'}]}}],
 'created': 1731393624,
 'id': 'chatcmpl-5bfb6d45-a1a8-4d93-b38a-82de34b0d533',
 'model': 'model/model-bf16-Q8_0.gguf',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 36, 'prompt_tokens': 491, 'total_tokens': 527}}
```


## Acknowledgement
--
본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다. (No. RS-2022-II220320, 상황인지 및 사용자 이해를 통한 인공지능 기반 1:1 복합대화 기술 개발)


