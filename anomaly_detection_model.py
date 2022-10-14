from sqlite3 import Timestamp


class MathematicalAnomalyDetection():

    """
    Anomaly detection based on simple mathematical operations.

    === Attributes ===
    benchmarks : Dict[str, float]  =>  dictionary containing various benchmarks that help 
                                                              with in detecting anomalies
    """

    def __init__(self):
        """
        Create new model with no benchmarks.
        """
        self.benchmarks = None

    def fit(self, gid:int, benchmarks_function:function):
        """
        gid: flow meter id
        benchmarks_function: function that returns a dictionary with useful benchmarks

        Initialize the benchmarks attribute with the appropriate benchmarks.
        """
        self.benchmarks = benchmarks_function(gid)

    def predict(self, flow:float, flow_prev:float, precipitation:float, hour:Timestamp):
        """
        Return wheter the given 'flow' value measurement is an anomaly or not, along with the anomaly label.

        return: 
            anomaly: 0 or 1
            anomaly_label: type of anomaly (in case it is an anomaly)
        """
        benchmarks = self.benchmarks

        # Average of (maximum difference in consecutive flow measurements in one day)
        avg_max_diff = benchmarks['avg_max_diff'].values[0]

        # Average standard deviation of 4 consecutive differences in flow measurements
        avg_sd_4 = benchmarks['mean_sd_4']

        # Average standard deviation of the last 4 measurements previous to [hour]
        # example: mean_sd_4_hour[15] = average standard deviation of flows at hours 14, 13, 12, 11.
        mean_sd_4_hour = benchmarks["mean_sd_4_hour"] 

        # Mean and standard deviation of all flow measurements grouped by hour 
        #   ^        ^
        mean_flow, sd_flow = benchmarks['mean_flow'], benchmarks['sd_flow']
        # example: mean_flow[4] = mean of all flow measurements recorded at 4AM.
        #          sd_flow[4] = standard deviation of all flow measurements recorded at 4AM.

        correctness = 1
        anomaly_label = None

        if not flow:
            correctness = 0
            anomaly_label = "no flow"

        if not flow_prev and flow_prev != 0: # Previous flow is NAN
            # If no data is available for the previous hour, check if current measurement is
            # within reasonable bounds for the hour that it has been recorded at.
            if (flow > mean_flow[hour] + 2*sd_flow[hour] or flow < mean_flow[hour] - 2*sd_flow[hour]):
                correctness = 0
                anomaly_label = "missing prev data & high deviation "
        
        else:
            curr_diff = flow - flow_prev # change in flow from previous flow to current flow
            if precipitation == False: # NO RAIN
                # If change in flow is too extreme.
                if abs(curr_diff) > max(2 * avg_sd_4, 2 * mean_sd_4_hour[hour], 2 * avg_max_diff):
                    anomaly_label = "[dry season] high deviation"
                    correctness = 0
                else:
                    correctness = 1
        
            elif precipitation > 0.0:
                pass
        # returns       anomaly, reason for anomaly
        return (1 - correctness, anomaly_label)
