import pyvisa
import numpy as np
import matplotlib.pyplot as plt

def map_range(x, a, b, c, d):
    return c + (x - a) * (d - c) / (b - a)

class TektronixDO:
    def __init__(self):
        self.resource = None
        self.instrumentName = None
        self.instrument = None
        self.time = None
        self.trace = None
        self.rm = pyvisa.ResourceManager()
        for _resource in self.rm.list_resources():
            _instrument = None
            try:
                _instrument = self.rm.open_resource(_resource)
            except:
                continue
            if _instrument != None:
                try:
                    _instrument.write("*IDN?")
                    response = _instrument.read()
                    if "TEKTRONIX,TBS 1102B" in response:
                        print("device found!")
                        self.instrumentName = response.replace("\n", "")
                        self.instrument = _instrument
                        self.resource = _resource
                        self.instrument.timeout = 2000
                        break
                    _instrument.close()
                except:
                    pass
    
    def cleanup(self) -> None:
        print("Stopping threads...")
        if self.instrument != None:
            self.instrument.close()
        if self.rm != None:
            self.rm.close()
    
    def get_trace(self) -> None:
        if self.instrument != None:
            self.instrument.write("data:encdg ascii")
            self.instrument.write("CURVE?")
            raw_data = self.instrument.read_raw()
            #print(type(raw_data))
            #print(f"Received {len(raw_data)} bytes")
            #print(raw_data)  # This will be in bytes format
            _data = [float(x) for x in raw_data.decode().split(',')] #[float(item.strip()) for item in raw_data.split(",")]
            # Normalize data
            self.trace = (_data-np.min(_data))/(np.max(_data)-np.min(_data))

            self.instrument.write("HORizontal?")
            response = self.instrument.read_raw()
            _data = [float(x) for x in response.decode().split(';')]
            N_t, scale_t, pos_t = int(_data[0]), _data[1], _data[2]
            self.time = np.linspace(0, scale_t*10, N_t) - (scale_t*10/2-pos_t)

            # self.instrument.write("MATH:VERtical?")
            # response = self.instrument.read_raw()
            # _data = [float(x) for x in response.decode().split(';')]
            # pos_v, scale_v  = _data[0], _data[1]
            # _min, _max = - (N_t/2*scale_v-pos_v), N_t*scale_v - (N_t/2*scale_v-pos_v)
            # print(np.min(self.trace), np.max(self.trace), _min, _max)
            # self.trace = map_range(self.trace, np.min(self.trace), np.max(self.trace), _min, _max)
            # print((_max-_min)/N_t)
            # self.instrument.write("DMA?")
            # response = self.instrument.read()
            # _data = [float(item.strip()) for item in response.splitlines() if item != ""]
            # print(_data)
            #plt.plot(_data)
            #plt.show()
        else:
            raise Exception("Instrument not found")

if __name__ == "__main__":
    DO = TektronixDO()
    DO.get_trace()
    DO.cleanup()
    print((DO.time[1]-DO.time[0])*1E6)
    plt.plot(DO.time, DO.trace)
    plt.show()