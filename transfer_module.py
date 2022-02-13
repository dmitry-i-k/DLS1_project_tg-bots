""" Использовался реализованный алгоритм из занятия.
    Создан класс с нейронной сетью.
    Вместе с функциями для реализации алгоритма помещены еще в один класс, реализованный в этом модуле
    """
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

#перенос стиля реализовал в классе
class Transfer_class():

    #клас со слоем нормализации
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
    #класс с сетью
    class Style_transfer(nn.Module):

        #для инициализации задается: обученная сеть(vgg16), нормализация, две картинке в виде тензоров
        #cpu/gpu, списки слоев, на которых берем лоссы, веса стиля и контента
        def __init__(self, cnn, norm_layer, style_img, content_img, device,
                     content_layers, style_layers,
                     style_weight=1_000_000, content_weight=1):
            super().__init__()
            self.style_weight = style_weight
            self.content_weight = content_weight
            self.style_img = style_img
            self.content_img = content_img
            self.content_layers = content_layers
            self.style_layers = style_layers
            self.device = device

            # определяем название последнего слоя сети, чтобы обрезать vgg
            loss_layers = self.content_layers + self.style_layers
            self.last_layer = max(loss_layers)
            cnn = copy.deepcopy(cnn)
            normalization = norm_layer.to(self.device)

            #первый слой - нормализация
            self.layers = nn.ModuleDict({'norm0': normalization})
            self.contents = {}
            self.styles = {}

            i = 0
            #пропускаем стиль и контент через его
            style = self.layers['norm0'](self.style_img)
            content = self.layers['norm0'](self.content_img)

            #идем по слоям исходной сети до последнего заданного
            for layer in cnn.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(i)
                    # Переопределим relu уровень
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(i)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(i)
                else:
                    raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
                #добавляем слои
                self.layers[name] = layer
                #пропускаем через них стиль и контент
                style = self.layers[name](style).detach()
                content = self.layers[name](content).detach()
                #если слои из списка - сохраняем пропущенные тензоры стиля и контента
                #из стиля считаем матрицу Грамма
                if name in content_layers:
                    self.contents[name] = content
                if name in style_layers:
                    self.styles[name] = self.gram_matrix(style)
                #если дошли до последнего заданного слоя, то сеть готова
                if name == self.last_layer:
                    break

        def forward(self, x):
            #будем собирать лоссы по заданным слоям
            content_losses = []
            style_losses = []

            for name, layer in self.layers.items():
                #пропускаем наше изображение
                x = layer(x)
                if name in self.content_layers:
                    # add content loss:
                    content_loss = F.mse_loss(x, self.contents[name])
                    content_losses.append(content_loss)
                if name in self.style_layers:
                    # add style loss:
                    G = self.gram_matrix(x)
                    style_loss = F.mse_loss(G, self.styles[name])
                    style_losses.append(style_loss)

            #сумируем лосы
            style_score = sum(style_losses)
            content_score = sum(content_losses)

            # взвешивание ошибки
            style_score *= self.style_weight
            content_score *= self.content_weight
            loss = style_score + content_score
            #возвращаем сумму взвешенных сум лоссов по заданным слоям
            return loss

        def gram_matrix(self, input):
            batch_size, h, w, f_map_num = input.size()  # batch size(=1)
            # b=number of feature maps
            # (h,w)=dimensions of a feature map (N=h*w)
            features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(batch_size * h * w * f_map_num)

    def __init__(self):
        #инициализируем device, тензоры для нормализации
        #списки слоев для подсчета лоссов
        #исходную обученную сеть
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

    #методы для загруузки файлов в  нужном формате
    def loader(self, imsize):
        return transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

    def image_loader(self, image_name, imsize):
        image = Image.open(image_name)
        image = self.loader(imsize)(image).unsqueeze(0)
        return image.to(self.device, torch.float)


    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    #основной метод  - оптимизируем изображение
    def run_mytransfer(self, content_img, style_img, input_img, num_steps=200,
                       style_weight=1_000_000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')


        model = self.Style_transfer(self.cnn, self.Normalization(self.normalization_mean, self.normalization_std),
                                    style_img, content_img, self.device,
                                    content_layers=self.content_layers_default,
                                    style_layers=self.style_layers_default,
                                    style_weight=style_weight, content_weight=content_weight)
        model.train()
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                loss = model(input_img)
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('loss:{:4f}'.format(loss.item()))
                return loss

            optimizer.step(closure)
        # a last correction...
        input_img.data.clamp_(0, 1)
        return input_img
    #метод принимает адреса исходных файлов, загружает их, запускает обработку изображения и выдает готовое изображение
    def __call__(self, style_fname, content_fname, style_level, resolution):
        style_img = self.image_loader(style_fname, resolution)
        content_img = self.image_loader(content_fname, resolution)
        input_img = content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        # input_img = torch.randn(content_img.data.size(), device=self.device)

        # add the original input image to the figure:
        # plt.figure()
        # imshow(input_img, title='Input Image')
        tensor = self.run_mytransfer(content_img, style_img, input_img, style_weight=style_level)

        output = tensor.cpu().clone()
        output = output.squeeze(0)
        unloader = transforms.ToPILImage()
        return unloader(output)