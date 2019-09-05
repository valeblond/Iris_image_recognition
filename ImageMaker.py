from PIL import Image, ImageDraw, ImageFont

# making matrix with given dimensions
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

#main class for image making
class ImageMaker:
    def __init__(self, nh):
        #general size of the picture
        self.size = 100
        #general size of neuron
        self.nsize = 4
        #number of hidden neurons
        self.nh = nh
        # coordinates of hidden neurons
        self.hiddencord = makeMatrix(nh, 2, 0.0)
        self.fontsize = self.size*2
        #variable that defines sensitivnes of weight change
        self.colorstrength = 1500
        #inistialazing hidden coordinates
        for i in range(nh):
            self.hiddencord[i][0] = self.nh * self.size * 10/2
            self.hiddencord[i][1] = (i + 1) * self.size*self.nsize*2
        # coordinates of inputs
        self.inputcord = makeMatrix(4, 2, 0.0)
        #coordinates of output neuron
        self.outputcord = makeMatrix(1, 2, 0.0)
        #initializing input coordinates
        for i in range(4):
            self.inputcord[i][0] = self.nh * self.size * 10/10
            self.inputcord[i][1] = self.hiddencord[0][1]/2 + self.size*self.nsize *(i+1) * 2
        # initializing output coordinates
        self.outputcord[0][0] = self.nh * self.size * 10 * 4/5
        self.outputcord[0][1] = round(nh * self.size * 10 / 2)
        #coordinates of input bias
        self.biasicord = makeMatrix(1, 2, 0.0)
        #coordinates of output bias
        self.biasocord = makeMatrix(1, 2, 0.0)
        #initializing bias coorinates
        self.biasicord[0][0] = self.nh * self.size * 10/5
        self.biasicord[0][1] = self.nh * self.size * 10+10
        self.biasocord[0][0] = round(nh * self.size * 10 * 3/5)
        self.biasocord[0][1] = self.nh * self.size * 10 + 10

    #function for drawing image
    def makeImage(self, input, doutput, poutput, ri, ro, k):
        im = Image.new('RGB', (self.nh * self.size * 10+10, self.nh * self.size * 10+10), color='white')
        draw = ImageDraw.Draw(im)
        r = 0
        g = 0
        #drawing hidden neurons
        for i in range(self.nh):
            if (ro[0][i]>0):
                g = round(ro[0][i]*self.colorstrength)
            else:
                r = -round(ro[0][i]*self.colorstrength)
            draw.ellipse([(self.hiddencord[i][0] - self.size * self.nsize/ 2, self.hiddencord[i][1] - self.size*self.nsize / 2),
                          (self.hiddencord[i][0] + self.size * self.nsize/ 2, self.hiddencord[i][1] + self.size * self.nsize / 2)],
                         (r, g, 0))
            r = 0
            g = 0
        #drowing output neuron
        if (poutput-doutput > 0):
            g = round((poutput-doutput) * self.colorstrength)
        else:
            r = -round((poutput-doutput) * self.colorstrength)
        draw.ellipse([(self.outputcord[0][0] - self.size * self.nsize/ 2, self.outputcord[0][1] - self.size*self.nsize / 2),
                          (self.outputcord[0][0] + self.size * self.nsize/ 2, self.outputcord[0][1] + self.size * self.nsize / 2)],
                         (r, g, 0))
        r = 0
        g = 0
        #writing input values
        font = ImageFont.truetype("arial.ttf", self.fontsize)
        for i in range(4):
            draw.text((self.inputcord[i][0]-self.nsize*2, self.inputcord[i][1]+self.nsize*2), str(input[i]), (125, 125, 125), font)

        #drawing synapses connecting inputs and hidden neurons
        for i in range(4):
            for j in range(self.nh):
                if (ri[j][i] > 0):
                    g = round(ri[j][i] * self.colorstrength)
                else:
                    r = -round(ri[j][i] * self.colorstrength)
                draw.line([(self.inputcord[i][0]+round(self.size)*2, self.inputcord[i][1]),
                           (self.hiddencord[j][0], self.hiddencord[j][1])],
                          (r, g, 0), 20)
                r = 0
                g = 0

        #drowing synapses connecting hiddel layer with output
        for j in range(self.nh):
            if (ri[j][4] > 0):
                g = round(ri[j][4] * self.colorstrength)
            else:
                r = -round(ri[j][4] * self.colorstrength)
            draw.line([(self.biasicord[0][0] + round(self.size), self.biasicord[0][1]),
                       (self.hiddencord[j][0], self.hiddencord[j][1])],
                      (r, g, 0), 20)
            r = 0
            g = 0

        #synapses for input bias
        for j in range(self.nh):
            if (ro[0][j] > 0):
                g = round(ro[0][j] * self.colorstrength)
            else:
                r = -round(ro[0][j] * self.colorstrength)
            draw.line([(self.hiddencord[j][0] + round(self.size), self.hiddencord[j][1]),
                        (self.outputcord[0][0], self.outputcord[0][1])],
                        (r, g, 0), 20)
            r = 0
            g = 0

        #synapses for output bias
        if (ro[0][self.nh] > 0):
            g = round(ro[0][self.nh] * self.colorstrength)
        else:
            r = -round(ro[0][self.nh] * self.colorstrength)
        draw.line([(self.biasocord[0][0] + round(self.size), self.biasocord[0][1]),
                   (self.outputcord[0][0], self.outputcord[0][1])],
                  (r, g, 0), 20)
        #writing ouput value
        draw.text((self.outputcord[0][0] + self.nsize * 2, self.outputcord[0][1] + self.nsize * 10), str(round(poutput,2))+ " : "+str(doutput),
                  (125, 125, 125), font)

        #saving image with given name
        im.save(str(k)+".png")