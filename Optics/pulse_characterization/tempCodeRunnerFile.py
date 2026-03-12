    _instrument.write("*IDN?")
                response = _instrument.read()
                if "TEKTRONIX,TBS 1102B" in response:
                    print("device found!")
                    self.instrumentName = response.replace("\n", "")
                    self.instrument = _instrument
                    self.resource = _resource
                    self.instrument.timeout = 5000
                    break
                _instrument.close()