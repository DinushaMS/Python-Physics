import pyvisa
import numpy as np
import matplotlib.pyplot as plt

class AnritsuOSA:
    def __init__(self):
        self.resource = None
        self.instrumentName = None
        self.instrument = None
        self.wlArray = None
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
                    if "ANRITSU,MS9710C" in response:
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
            self.instrument.write("DCA?")
            # Read raw bytes from the instrument
            response = self.instrument.read()
            _data = [float(item.strip()) for item in response.split(",")]
            _startWL, _stopWL, _count = _data[0], _data[1], _data[2]
            self.wlArray = np.linspace(_startWL, _stopWL, int(_count))

            self.instrument.write("DMA?")
            response = self.instrument.read()
            _data = [float(item.strip()) for item in response.splitlines() if item != ""]
            self.trace = _data
        else:
            raise Exception("Instrument not found")

if __name__ == "__main__":
    OSA = AnritsuOSA()
    OSA.get_trace()
    OSA.cleanup()