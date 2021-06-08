class Utils:
    @staticmethod
    def cast_int(s):
        try:
            return int(s)
        except:
            return None

    @staticmethod
    def cast_float(s):
        try:
            return float(s)
        except:
            return None
