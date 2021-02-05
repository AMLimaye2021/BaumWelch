import sys, numpy


class hiddenPath:
    '''
        Class to create a set of matrices for soft decoding and executing the Baum Welch algorithm on a set of data
        provided by the user


        __init__ method is the constructor for the nodeGraph
        Args:
            seq (string): the text string of the sequence
            emissionMat (matrix): emission matrix of probabilities
            emissionKey (dictionary): dictionary of associated indexes of emission matrix
            transMat (matrix): transition matrix of probabilities
            transKey (dictionary): dictionary of associated indexes of transition matrix
            iterations (int): the number of iterations for the Baum Welch Algorithm

        Attributes:
            seq (string): the text string of the sequence
            emissionMat (matrix): emission matrix of probabilities
            emissionKey (dictionary): dictionary of associated indexes of emission matrix
            transMat (matrix): transition matrix of probabilities
            transKey(dictionary): dictionary of associated indexes of transition matrix
            startProb (matrix): transition matrix meant for the transition from the start node to the first state, filled
                               with the same value of 1/total states
            forwardDiagram (matrix): stores the forward probabilities of the Viterbi Diagram
            backwardDiagram (matrix): stores the backward probabilities of the Viterbi Diagram
            forwardWeight (float): the forward weight of the sequence provided
            edgeResponsibility (list of matrices): Edge responsibility matrix for each state at each time i
            probDiagram (matrix): probability of being in a state at time i (solution to the soft decoding problem)
            iterations (int): the number of iterations for the Baum Welch Algorithm
    '''
    def __init__(self):
        self.seq = ''
        self.revSeq = ''
        self.emissionMat = None
        self.emissionKey = {}
        self.transMat = None
        self.transKey = {}
        self.startProb = None
        self.forwardDiagram = None
        self.backwardDiagram = None
        self.forwardWeight = 0.0
        self.edgeResponsibility = []
        self.probDiagram = None
        self.iterations = 0

    def readInput(self):
        '''
            This function reads in the values from the input file and initializes the attributes of the class

            Args:
                N/A

            Attributes:
                N/A

        '''
        file = sys.stdin
        iterations = file.readline().strip('\n')
        self.iterations = int(iterations)
        file.readline()
        self.seq = file.readline().strip('\n')
        self.revSeq = self.seq[::-1]
        trash = file.readline()
        l = file.readline()
        l = l.split()
        trash = file.readline()
        stateL = file.readline()
        stateL = stateL.split()

        for i in range(0, 2):
            trash = file.readline()

        transMat = []

        while True:
            i = file.readline()
            temp = i.split()
            num = []
            if temp[0] not in stateL:
                break
            for j in temp[1:len(temp)]:
                num.append(numpy.float_(j))
            transMat.append(num)
        self.transMat = numpy.array(transMat)
        trash = file.readline()

        emissionMat = []
        for i in file.readlines():
            temp = i.split()
            num = []
            for j in temp[1:len(temp)]:
                num.append(numpy.float_(j))
            emissionMat.append(num)
        self.emissionMat = numpy.array(emissionMat)

        index = 0
        emissionKey = {}
        for i in l:
            emissionKey[i] = index
            index += 1
        self.emissionKey = emissionKey

        key = 0
        transKey = {}
        for i in stateL:
            transKey[i] = key
            key += 1
        self.transKey = transKey
        self.startProb = numpy.full(len(self.transMat), 1 / len(self.transMat))
        self.forwardDiagram = numpy.empty((len(self.transMat), len(self.seq)))
        self.backwardDiagram = numpy.empty((len(self.transMat), len(self.seq)))
        self.probDiagram = numpy.empty((len(self.transMat), len(self.seq)))

    def setStep(self, diagram, node):
        '''
            Executes the dot product of the viterbi diagram till the current node and sums the probabilities to that
            point for each state

            Args:
                diagram (matrix): the viterbi diagram at the present time of the passed in node
                node (int): the current time state which is tied to a series of nodes in the viterbi diagram

            Returns:
                prob (matrix): this matrix contains the probability for all states
        '''
        prob = numpy.sum(diagram[:, node - 1] * self.transMat.T * self.emissionMat[numpy.newaxis, :, self.emissionKey.get(self.seq[node])].T, 1)
        return prob

    def setReverse(self, diagram, node):
        '''
            Executes the dot product of the viterbi diagram till the current node and sums the probabilities to that
            point for each state but in the backwards direction.

            Args:
                diagram (matrix): the viterbi diagram at the present time of the passed in node
                node (int): the current time state which is tied to a series of nodes in the viterbi diagram

            Returns:
                prob (matrix): this matrix contains the probability for all states
        '''
        dmat = diagram[:, node + 1]
        etmat = self.transMat.T * self.emissionMat[numpy.newaxis, :, self.emissionKey.get(self.seq[node + 1])].T
        einmat = numpy.einsum('ij,i->ij', etmat, dmat)
        prob = numpy.sum(einmat, 0)
        return prob

    def findOutcomeLikelihood(self):
        '''
            Given a sequence, emission matrix of probabilities, and transition matrix of probabilities, calculate the
            forward and backwards diagrams of state probability as well as the maximum forwards weight of the sequence

            Args:
                N/A

            Returns:
                N/A
        '''

        matrixLength = len(self.transMat)  # length of the matrix
        length = len(self.seq)  # length of the sequence
        viterbiDiagram = numpy.empty((matrixLength, length))  # create the empty viterbi diagram of size [states, sequence length]
        revViterbiDiagram = numpy.empty((matrixLength, length))
        viterbiDiagram[:, 0] = self.startProb * self.emissionMat[:, self.emissionKey.get(self.seq[0])]
        viterbiDiagram[:, 0] = self.emissionMat[:, self.emissionKey.get(self.seq[0])]# initialize the first values of viterbi diagram
        revViterbiDiagram[:, length - 1] = 1  # initialize the first values of viterbi diagram


        for i in range(1, length): # for every time/node set i, calculate the maximum probability and associated state
            viterbiDiagram[:, i] = self.setStep(viterbiDiagram, i) # update the viterbi diagram with the probabilities calculated

        for i in range(length - 2, -1, -1):
            revViterbiDiagram[:, i] = self.setReverse(revViterbiDiagram, i)

        terminus = numpy.empty((matrixLength))
        for i in range(0, matrixLength):
            terminus[i] = revViterbiDiagram[i, 0] * self.emissionMat[i, self.emissionKey.get(self.seq[0])]

        z = numpy.sum(viterbiDiagram[:, length - 1], dtype=numpy.float_) # calculates the final forward weight
        f = numpy.sum(terminus, dtype=numpy.float_)
        self.forwardDiagram = viterbiDiagram
        self.backwardDiagram = revViterbiDiagram
        self.forwardWeight = z

    def softDecoding(self):
        '''
            Given the forwards and backwards diagrams of weights for the sequence, determine the probability of being
            in each of the states at time i

            Args:
                N/A

            Returns:
                N/A

        '''

        matrixLength = len(self.transMat)  # length of the matrix
        length = len(self.seq)  # length of the sequence
        probDiagram = numpy.empty((matrixLength, length))  # create the empty diagram of previous states at each node

        for i in range(0, length):  # for every time/node set i, calculate the maximum probability and associated state
            for j in range(0, len(self.startProb)):
                f = self.forwardDiagram[j, i]
                b = self.backwardDiagram[j, i]
                w = self.forwardWeight
                probDiagram[j, i] = (self.forwardDiagram[j, i] * self.backwardDiagram[j, i])/self.forwardWeight
        self.probDiagram = probDiagram

    def setEdge(self, diagram, node):
        '''
            Executes the dot product of the viterbi diagram till the current node and creates a matrix of the
            probabilities for all edges at time i

            Args:
                diagram (matrix): the viterbi diagram at the present time of the passed in node
                node (int): the current time state which is tied to a series of nodes in the viterbi diagram

            Returns:
                einmat (matrix): the probability of any edge being traversed at time i
        '''
        f = self.forwardDiagram[:, node]
        b = self.backwardDiagram[:, node + 1]
        et = self.transMat.T * self.emissionMat[numpy.newaxis, :, self.emissionKey.get(self.seq[node + 1])].T
        net = et * (1/self.forwardWeight)

        einmat = numpy.einsum('ij,i,j->ij', net, b, f)
        e = einmat[:,0]
        return einmat

    def getEdgeResponsibility(self):
        '''
            Given the forwards and backwards diagrams of weights for the sequence, create a matrix of all edge
            responsibilities

            Args:
                N/A

            Returns:
                N/A

        '''
        matrixLength = len(self.transMat)  # length of the matrix
        length = len(self.seq)  # length of the sequence

        # for i in range(0, matrixLength):
        erDiagram = numpy.empty((matrixLength, length))
        for j in range(0, length-1):
            erDiagram = self.setEdge(erDiagram, j)
            self.edgeResponsibility.append(erDiagram)

    def redefineParams(self):
        '''
            Given the probability matrix (pi*) and edgeResponsibility matrix (pi**), redefine the parameters of the HMM
            by using the matrices to calculate a new set of parameter matrices.

            Args:
                N/A

            Returns:
                N/A

        '''
        temp = numpy.empty((len(self.transKey), len(self.emissionKey)))
        emissionMatrix = numpy.empty((len(self.transKey), len(self.emissionKey)))
        temp = temp * 0
        for i in range(0, len(self.seq)):
            for j in range(0, len(self.transKey)):
                index = self.emissionKey.get(self.seq[i])
                temp[j, index] = temp[j, index] + self.probDiagram[j,i]

        sumMatrix = numpy.empty((len(self.transKey)))
        for i in range(0, len(self.transKey)):
            test = temp[i]
            sumMatrix[i] = 1/numpy.sum(temp[i], 0)

        emissionMatrix = numpy.einsum('ij,i->ij', temp, sumMatrix)

        total = numpy.empty((len(self.transKey), len(self.transKey)))
        total = total * 0
        for i in range(0, len(self.edgeResponsibility)):
            e = self.edgeResponsibility[i].T
            total = total + self.edgeResponsibility[i].T
        sumTotal = 1/numpy.sum(total, 1)

        transitionMatrix = numpy.empty((len(self.transKey), len(self.transKey)))
        transitionMatrix = numpy.einsum('ij,i->ij', total, sumTotal)

        self.emissionMat = emissionMatrix
        self.transMat = transitionMatrix

    def baumWelch(self):
        '''
            Execute the E step then M step of the Baum Welch algorithm in sequence for the amount of time specified
            by the user in Iterations. At the end of this loop, output the new parameters given by the alogrithm.

            Args:
                N/A

            Returns:
                N/A

        '''
        for i in range(0, self.iterations):
            self.findOutcomeLikelihood()
            self.softDecoding()
            self.getEdgeResponsibility()
            self.redefineParams()

        print('Transition Matrix')
        print(self.transMat)
        print('\n', 'Emission Matrix')
        print(self.emissionMat)

def main():
    myPath = hiddenPath()
    myPath.readInput()
    myPath.baumWelch()


if __name__ == "__main__":
    main()