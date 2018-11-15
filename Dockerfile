FROM python:3.6

ADD . /agent

RUN pip install gym
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN cd /agent/playground/ && pip install .
RUN pip install opencv-python
RUN pip install matplotlib

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent/source
ENTRYPOINT ["python"]
CMD ["run.py"]
