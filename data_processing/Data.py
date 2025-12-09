class Data():
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.label = label
        self.sampling_frequency = sampling_frequency
        self.f_mains = f_mains
    
    def get_points_per_cycle(self):
        return int(self.sampling_frequency / self.f_mains)
    
    def get_time_step(self):
        return 1 / self.sampling_frequency
    
    def get_label(self):
        return self.label