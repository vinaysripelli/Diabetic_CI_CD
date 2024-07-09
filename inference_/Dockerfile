FROM python:3.10
WORKDIR /inference
EXPOSE 8089
COPY . .
COPY requirements.txt /inference
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8089"]