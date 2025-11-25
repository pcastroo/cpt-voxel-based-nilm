class Data():
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.label = label
        self.sampling_frequency = sampling_frequency
        self.f_mains = f_mains

        self.is_underrepresented = False # flag for underrepresented classes

    @staticmethod
    def check_underrepresented(data_list, min_samples=50):  # check which classes are underrepresented
        import numpy as np
        
        all_labels = [data.label for data in data_list]
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        
        underrepresented_classes = set()
        for cls, count in zip(unique_classes, class_counts):
            if count < min_samples:
                underrepresented_classes.add(cls)
        
        if underrepresented_classes:
            for data in data_list:
                if data.label in underrepresented_classes:
                    data.is_underrepresented = True
    
    def get_points_per_cycle(self):
        return int(self.sampling_frequency / self.f_mains)
    
    def get_time_step(self):
        return 1 / self.sampling_frequency