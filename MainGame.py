import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pygame
import sys
from PIL import Image

pygame.init()
pygame.font.init() 

# Define a simple CNN model for image classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # 1 input channel (grayscale), 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3) # 16 input channels, 32 output channels, 3x3 kernel
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)pip show torch
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)  # Max pooling with a 2x2 window
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 5 * 5)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PytorchWorkFlow():
    # Load and preprocess the MNIST dataset
    def __init__(self):
        self.dataRootPath = './data'
        self.modelName = 'new_model.pth'
        self.modelPath = os.path.join(self.dataRootPath, self.modelName)
        self.transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to (-1, 1)
        ])

        # Load MNIST training and test datasets
        self.trainset = torchvision.datasets.MNIST(root=self.dataRootPath, train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=32, shuffle=True)

        self.testset = torchvision.datasets.MNIST(root=self.dataRootPath, train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)


        # Initialize the CNN, loss function, and optimizer
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # Model is here
        self.epochs = 3  # Number of epochs for training

        if os.path.exists(self.modelPath):
            print("Model file found. Loading the model...")
            self.model.load_state_dict(torch.load(self.modelPath))
            self.model.eval()  # Set the model to evaluation mode
        else:
            print("Model file not found. Training the model...")
            # self.Train()

        # Train the model


    def Train(self):
        if self.model.training:
            for epoch in range(self.epochs):
                running_loss = 0.0
                for i, data in enumerate(self.trainloader, 0):
                    # i is the batch index, starting from 0
                    # data is a tuple (inputs, labels) for the current batch
                    inputs, labels = data

                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    if i % 200 == 199:  # Print loss every 200 batches
                        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                        running_loss = 0.0
            #Save model here
            torch.save(self.model.state_dict(), self.modelPath) 
            print("Finished Training")

    def GeneralTest(self):
        # Evaluate the model on the test dataset
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Create a formatted string variable
        accuracy_message = f'Accuracy of the model on the 10,000 test images: {accuracy:.2f}%'

        # Print the variable
        return accuracy_message
    
    def BatchTest(self,batch_index=0):
        # Get a batch of test images
        # dataiter = iter(self.testloader)
        # images, labels = next(dataiter)
        # Print the model's predictions

        all_batches = list(self.testloader)
        # print(len(all_batches))
        
        # Ensure the batch_index is within range
        if batch_index < 0 or batch_index >= len(all_batches):
            raise IndexError("Batch index out of range")

        # Select the desired batch
        images, labels = all_batches[batch_index]


        outputs = self.model(images)
        _, predicted = torch.max(outputs, 1)

        predictedText = 'Predicted: '+' '.join(f'{predicted[j].item()}' for j in range(len(predicted)))
        actualText = 'Actual:    '+' '.join(f'{labels[j].item()}' for j in range(len(labels)))

        # Show images
        return [torchvision.utils.make_grid(images),predictedText,actualText]

    # Visualize a few test images along with the model's predictions
    def imshow(self,img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


class RayChengPyTorchGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("RayCheng PyTorch")
        self.mouse_pos = pygame.mouse.get_pos()
        self.isUserTesting = False

        self.button_color = (40, 40, 40)  
        self.button_hover_color = (80, 80, 80)  
        self.button_click_color = (120, 120, 120) 
        self.button_batch = Button(button_x = self.WIDTH// 2, 
                button_y = self.HEIGHT - 80,
                button_width =300, 
                button_height = 70,
                button_normal_color = self.button_color,
                button_hover_color = self.button_hover_color,
                button_click_color = self.button_click_color,
                button_text='Show Next Batch',
                action= self.show_next_batch)
        self.button_usertest = Button(button_x = self.WIDTH  // 2, 
                button_y = self.HEIGHT - 160,
                button_width =300, 
                button_height = 70,
                button_normal_color= self.button_color,
                button_hover_color = self.button_hover_color,
                button_click_color = self.button_click_color,
                button_text='Draw your input',
                action= self.trigger_user_input_mode)
        self.button_confirm = Button(button_x = self.WIDTH  // 2 -80, 
                button_y = self.HEIGHT - 160,
                button_width =140, 
                button_height = 70,
                button_normal_color= (0,120,0),
                button_hover_color = (0,160,0),
                button_click_color = self.button_click_color,
                button_text='Confirm',
                action= self.click_confirm_button)

        self.button_reset = Button(button_x = self.WIDTH  // 2 +80, 
                button_y = self.HEIGHT - 160,
                button_width =140, 
                button_height = 70,
                button_normal_color= (120,0,0),
                button_hover_color = (160,0,0),
                button_click_color = self.button_click_color,
                button_text='Reset',
                action= self.click_reset_button)


        #Workflow for PyTorch data
        self.workflow = PytorchWorkFlow()
        self.imgTorch = None
        self.batchSize = len(list(self.workflow.testloader))
        self.batchIndex = 0
        self.predictedResultText = ''
        self.actualResultText = ''
        self.generalAcuracyText=""
        self.userTestText=""

        #User drawing functions
        self.brush_radius = 5
        self.drawing_area_length = 250
        self.drawing_area_color = (140,140,140)
        self.drawing_area = DrawningRegion(x = (self.WIDTH-self.drawing_area_length)//2,
                                           y = (self.HEIGHT-self.drawing_area_length)//2, 
                                           width = self.drawing_area_length, 
                                           height = self.drawing_area_length,
                                           color = self.drawing_area_color,
                                           screen = self.screen,
                                           brush_radius = self.brush_radius)
        
    def click_confirm_button(self):
        userImg = self.drawing_area.save_img()
        outputs = self.workflow.model(userImg)
        _, predicted = torch.max(outputs, 1)
        print(predicted[0].item())
        self.userTestText="The result is: "+str(predicted[0].item())


    def click_reset_button(self):
        self.drawing_area.drawing_surface.fill((0, 0, 0)) 
        self.userTestText=""
    def trigger_user_input_mode(self):
        self.isUserTesting = True

    def show_next_batch(self):
        self.isUserTesting = False
        self.imgTorch,self.predictedResultText, self.actualResultText = self.workflow.BatchTest(self.batchIndex)
        self.batchIndex = (self.batchIndex + 1) % self.batchSize
        


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.button_batch.check_click(self.mouse_pos)
                if(self.isUserTesting):
                    
                    self.drawing_area.check_mouse_down(self.mouse_pos)
                    self.button_confirm.check_click(self.mouse_pos)
                    self.button_reset.check_click(self.mouse_pos)
                else:
                    self.button_usertest.check_click(self.mouse_pos)
                

    def update(self):
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        self.mouse_pos = pygame.mouse.get_pos()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
        if(self.isUserTesting):
            # Check if the left mouse button is pressed
            if mouse_buttons[0]:
                self.drawing_area.check_mouse_down(self.mouse_pos)
            else:
                self.drawing_area.last_mouse_pos = None



    def drawScreen(self):
        # Fill the screen with white
        self.screen.fill((0, 0, 0))
        self.draw_text(self.screen,'RayCheng PyTorch',64,self.WIDTH/2,50)
        self.button_batch.draw(self.screen,self.mouse_pos)
        # print(self.isUserTesting)
        if (self.isUserTesting):
            self.drawing_area.refresh_on_screen()
            self.drawing_area.mouse_move_in(self.mouse_pos)
            self.button_confirm.draw(self.screen,self.mouse_pos)
            self.button_reset.draw(self.screen,self.mouse_pos)
            self.draw_text(self.screen,self.userTestText,50,self.WIDTH/2,125,color=(255,255,0))
        else:
            self.button_usertest.draw(self.screen,self.mouse_pos)
            self.pytorch_imgshow_pygame(self.imgTorch, self.screen, x=self.WIDTH//2, y=self.HEIGHT//2)
            self.draw_text(self.screen,self.predictedResultText,30,self.WIDTH/2,150)
            self.draw_text(self.screen,self.actualResultText,30,self.WIDTH/2,200)


        # Update the display
        pygame.display.flip()

    def draw_text(self,surf,text,size,x,y,color=(255,255,255)):
        font=pygame.font.Font(None,size)
        text_surface=font.render(text,True,color)
        text_rect = text_surface.get_rect()
        text_rect.midtop =(x,y)
        surf.blit(text_surface,text_rect)
    
    def pytorch_imgshow_pygame(self, img, screen, x=0, y=0):
        if(img!=None):
            img = img / 2 + 0.5  # Unnormalize the image
            npimg = img.numpy()

            # Transpose the image from (C, H, W) to (H, W, C) for Pygame (height, width, channels)
            npimg = np.transpose(npimg, (2, 1, 0))
            # npimg = np.flipud(npimg)  # Flip the image vertically
            # npimg = np.fliplr(npimg)

            # Convert the image to a Pygame surface
            npimg = np.clip(npimg * 255, 0, 255).astype(np.uint8)  # Scale the image to 0-255 range
            surface = pygame.surfarray.make_surface(npimg)
            

            # Blit the surface onto the screen at the specified position
            screen.blit(surface, (x-surface.get_width()//2, y-surface.get_height()//2))

    def run(self):
        # Main game loop
        while True:
            self.drawScreen()
            self.handle_events()
            self.update()
            # Frame rate
            pygame.time.Clock().tick(30)

class DrawningRegion(object):
    def __init__(self, x, y, width, height,color,screen,brush_radius):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.screen = screen
        self.radius = brush_radius
        self.brush = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.brush, (255, 255, 255), (self.radius, self.radius), self.radius)

        self.drawing_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.drawing_surface.fill((0, 0, 0)) 

        self.last_mouse_pos = None
        


    def mouse_move_in(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            brush_rect = self.brush.get_rect(center=(mouse_pos[0], mouse_pos[1]))
            self.screen.blit(self.brush, brush_rect)
            

    def check_mouse_down(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            # print(self.last_mouse_pos, mouse_pos)
            local_mouse_pos = (mouse_pos[0] - self.rect.x, mouse_pos[1] - self.rect.y)
            brush_rect = self.brush.get_rect(center=local_mouse_pos)

            if self.last_mouse_pos:
                pygame.draw.line(self.drawing_surface, (255, 255, 255), self.last_mouse_pos, local_mouse_pos, self.radius * 2)
            else:
                self.drawing_surface.blit(self.brush, brush_rect)


            self.last_mouse_pos = local_mouse_pos
            


    def refresh_on_screen(self): 
        # Blit the drawing surface to the main screen at the position of the drawing region
        self.screen.blit(self.drawing_surface, self.rect.topleft)
        # Draw the border of the drawing region on the main screen
        pygame.draw.rect(self.screen, self.color, self.rect, 2)

    def save_img(self):
        # Assuming 'drawing_surface' is your surface
        surface_array = pygame.surfarray.array3d( self.drawing_surface)  # Convert surface to a 3D array (RGB)

        # Convert to grayscale for MNIST
        grayscale_image = np.mean(surface_array, axis=2).astype(np.uint8)  # Average the RGB values to get grayscale
        pil_image = Image.fromarray(grayscale_image)
        resized_image = pil_image.resize((28, 28))
        mnist_image = np.array(resized_image).reshape(28, 28)
        mnist_image = mnist_image / 255.0 
        mnist_image = np.transpose(mnist_image, (1, 0))

        # plt.imshow(mnist_image, cmap='gray')
        # plt.title('Preview Image')
        # plt.show()

        mnist_tensor = torch.tensor(mnist_image).unsqueeze(0).unsqueeze(0).float()
        return mnist_tensor



class Button:
    def __init__(self,
                button_x = 0, 
                button_y = 0,
                button_width =200, 
                button_height = 100,
                button_text='Button',
                button_font = pygame.font.Font(None, 24),
                button_normal_color=(40, 40, 40),
                button_hover_color=(80, 80, 80),
                button_click_color=(120, 120, 120),
                button_text_color=(255, 255, 255),
                action=None):
        

        self.rect = pygame.Rect(button_x, button_y, button_width, button_height)
        self.rect.midtop =(button_x,button_y)
        self.button_normal_color = button_normal_color
        self.button_hover_color = button_hover_color
        self.button_click_color = button_click_color
        self.text = button_text
        self.action = action  # Function to call when button is pressed
        self.font = button_font
        self.button_text_color = button_text_color

    def draw(self, screen,mouse_pos):
        
        if self.rect.collidepoint(mouse_pos):  # Mouse is hovering over the button
            curColor = self.button_hover_color
        else:
            curColor = self.button_normal_color
        pygame.draw.rect(screen, curColor, self.rect)
        text_surface = self.font.render(self.text, True, self.button_text_color)
        screen.blit(text_surface, (self.rect.x + (self.rect.width-text_surface.get_width())//2, self.rect.y + (self.rect.height-text_surface.get_height())//2))

    def check_click(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            if self.action:
                self.action()


# Main function to run the game

game = RayChengPyTorchGame()
game.run()








