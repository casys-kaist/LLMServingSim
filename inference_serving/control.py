import re

class Controller():
    def __init__(self, total_num, verbose=False):
        self.end_dict = {}
        self.total_num = total_num
        self.verbose = verbose
        for i in range(total_num):
            self.end_dict[i] = -1


    def readWait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        return out

    def checkEnd(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        for i in out[4:]:
            print(i, end='')
        return out

    def writeFlush(self, p, input):
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parseOutput(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles"
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))

            if self.end_dict[sys] != id:
                if self.verbose:
                    trimmed_output = output[output.find("sys["):]
                    print('Control: ' + trimmed_output, end='')
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return