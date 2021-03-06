import grpc

# classes generated by grpc
from .generated import eval_server_pb2
from .generated import eval_server_pb2_grpc

# other imports
import numpy as np 
import time
from time import perf_counter
from datetime import datetime

# multiprocessing 
import multiprocessing as mp
from multiprocessing import shared_memory, resource_tracker
from threading import Thread

# suppress shared memory warnings
from .utils import remove_shm_from_resource_tracker
remove_shm_from_resource_tracker()

# receive input fidx streamed by server, store them in a list
def receive_stream(seq, latest_fidx, fid_ptr_dict, is_stream_ready, stream_start_time, config, verbose=False):
    if verbose:
        print("EvalClient (", datetime.now(), "): ", "Requesting stream for sequence ", seq)
    channel = grpc.insecure_channel(config['loopback_ip'] + ":" + str(config['image_service_port']))
    stub = eval_server_pb2_grpc.ImageServiceStub(channel)
    stream_request = eval_server_pb2.String(value=seq)
    send_times = []

    # receive input stream
    for i, response in enumerate(stub.GetImageStream(stream_request)):
        if i == 0:
            stream_start_time.value = perf_counter()
            if verbose:
                print("EvalClient (", datetime.now(), "): ", "Receiving stream for sequence ", seq) 
        if response.end_marker:
            latest_fidx.value = -1
            break
        is_stream_ready.clear()
        latest_fidx.value = response.fid
        fid_ptr_dict[response.fid] = (response.start_ptr, response.end_ptr)
        if response.fid >= 0:
            is_stream_ready.set()
        send_times.append(perf_counter() - response.timestamp)

    # if verbose:
    #     print("EvalClient (", datetime.now(), "): ", "Mean sending time = ", np.mean(send_times), "s, stdev = ", np.std(send_times), "Max = ", np.max(send_times), "Min = ", np.min(send_times))

    channel.close()

class EvalClient:

    def __init__(self, config, state=None, verbose=False):
        self.img_width, self.img_height = 1920, 1200

        if state is None:
            mp.set_start_method('spawn')
            self.latest_fidx = mp.Value('i', -1, lock=True)
            self.is_stream_ready = mp.Event()
            self.fid_ptr_dict = mp.Manager().dict()
            self.stream_start_time = mp.Value('d', 0.0, lock=True)
        else:
            self.latest_fidx = state[0]
            self.is_stream_ready = state[1]
            self.fid_ptr_dict = state[2]

        
        self.verbose = verbose
        # create image receiver stub
        self.channel = grpc.insecure_channel(config['loopback_ip'] + ":" + str(config['image_service_port']))
        self.config = config
        self.stub = eval_server_pb2_grpc.ImageServiceStub(self.channel)
        response = self.stub.GetShm(eval_server_pb2.Empty())

        self.existing_shm = shared_memory.SharedMemory(name=response.value)
        self.channel.close()

        # create result sender stub
        self.result_channel = grpc.insecure_channel(config['loopback_ip'] + ":" + str(config['result_service_port']))
        self.result_stub = eval_server_pb2_grpc.ResultServiceStub(self.result_channel)

        response = self.result_stub.GetShm(eval_server_pb2.Empty())
        self.results_shm = shared_memory.SharedMemory(name=response.value)
        self.results_np = np.ndarray((100, 6), dtype=np.float32, buffer=self.results_shm.buf)

        self.is_stream_ready.clear()
        self.stream_process = None

        self.result_thread = None

    def get_state(self):
        return (self.latest_fidx, self.is_stream_ready, self.fid_ptr_dict)

    def close(self, results_file='results.json'):
        self.result_stub.GenResults(eval_server_pb2.String(value=results_file))
        self.result_channel.close()
        self.existing_shm.close()
        self.results_shm.close()

    def stop_stream(self):
        self.stream_process.join()
        self.result_stub.FinishSequence(eval_server_pb2.Empty())
        self.is_stream_ready.clear()
        self.stream_process = None

    def request_stream(self, seq):
        # receiver as separate processs
        self.stream_process = mp.Process(target=receive_stream, args=(seq, self.latest_fidx, self.fid_ptr_dict, self.is_stream_ready, self.stream_start_time, self.config, self.verbose))
        self.stream_process.start()

    def get_latest_fidx(self):
        self.is_stream_ready.wait()
        return self.latest_fidx.value

    def get_frame(self, fid=None, ptr=False):
        if fid is not None and fid < 0:
            raise TypeError(f"fid must be non-negative")
        if fid is None:
            fid = self.get_latest_fidx()
            if fid == -1:
                return None, None
        elif fid not in self.fid_ptr_dict:
            raise KeyError(f"frame not available yet")
        start_ptr, end_ptr = self.fid_ptr_dict[fid]
        if ptr:
            return fid, int(start_ptr/(self.img_height*self.img_width*3))
        return fid, np.ndarray((self.img_height, self.img_width, 3), dtype=np.uint8, buffer=self.existing_shm.buf[start_ptr:end_ptr])

    def send_result_shm(self, bboxes, bbox_scores, labels, timestamp):
        num_detections = min(len(bboxes), 100)
        self.results_np[:num_detections, :4] = bboxes[:num_detections]
        self.results_np[:num_detections, 4] = bbox_scores[:num_detections]
        self.results_np[:num_detections, 5] = labels[:num_detections]
        self.result_stub.SignalResultsReady(eval_server_pb2.Result(timestamp=timestamp, num_bboxes=num_detections))

    def send_result_to_server(self, bboxes, bbox_scores, labels):
        timestamp = perf_counter()
        if self.result_thread:
            self.result_thread.join()
        self.result_thread = Thread(target=self.send_result_shm, args=(bboxes, bbox_scores, labels, timestamp))
        self.result_thread.start()

    def get_frame_buf(self):
        return self.existing_shm

    def get_stream_start_time(self):
        self.is_stream_ready.wait()
        return self.stream_start_time.value