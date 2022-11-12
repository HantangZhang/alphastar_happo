import flock, os, torch

class gpu_share_unit():
    def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party=''):
        self.device = which_gpu
        self.manual_gpu_ctl = True
        self.lock_path=lock_path
        self.gpu_party = gpu_party
        if gpu_party == 'off':
            self.manual_gpu_ctl = False
        
        if self.lock_path is None: 
            self.lock_path=os.path.dirname(__file__)

    def __enter__(self):
        self.get_gpu_lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_gpu_lock()

    def get_gpu_lock(self):
        if self.manual_gpu_ctl:
            print('wait for gpu...', end='', flush=True)
            self.gpu_lock_file = None
            self.gpu_lock = None
            # print(self.lock_path+'/lock_gpu_%s_%s'%(self.device, self.gpu_party))
            self.gpu_lock_file = open(self.lock_path+'/lock_gpu_%s_%s.glock'%(self.device, self.gpu_party), 'w+')
            self.gpu_lock = flock.Flock(self.gpu_lock_file, flock.LOCK_EX)
            self.gpu_lock.__enter__()
            print('get!')
        return

    def release_gpu_lock(self):
        if self.manual_gpu_ctl:
            torch.cuda.empty_cache()
            self.gpu_lock.__exit__(None,None,None)
            self.gpu_lock_file.close()
        return