# 1. 베이스 이미지 선택
FROM python:3.10

# 2. 작업 디렉토리 생성 및 설정
WORKDIR /home/my_project

# 3. 필수 패키지 복사 및 설치
COPY requirements.txt ./ 
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 4. Git 설치 및 불필요한 파일 제거
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. 컨테이너 실행 시 명령어 설정
CMD ["bash"]