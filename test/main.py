import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import datetime

# 데이터셋 및 변환 설정
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 데이터셋 경로 설정
data_dir = 'images'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'

# 데이터 로딩 설정
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 모델 생성 및 손실 함수, 최적화 알고리즘 설정
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(512, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 대시보드 생성
app = dash.Dash(__name__)

# 그래프 초기화
loss_data = {'train': [], 'test': []}
accuracy_data = []

# 학습 함수
def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 로그 저장
        loss_data['train'].append(loss.item())

    epoch_loss = running_loss / len(dataloader)

    # 로그 파일에 저장
    log_string = f"Epoch {epoch}/{num_epochs} Train Loss: {epoch_loss:.4f}\n"
    with open(log_file, 'a') as f:
        f.write(log_string)

    return epoch_loss


# 평가 함수
def evaluate(model, dataloader, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 로그 저장
            loss_data['test'].append(loss.item())
            accuracy_data.append(correct / total)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    # 로그 파일에 저장
    log_string = f"Epoch {epoch}/{num_epochs} Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n"
    with open(log_file, 'a') as f:
        f.write(log_string)

    return epoch_loss, accuracy


# 학습 및 평가 수행
num_epochs = 10
epoch_results = []

# 로그 파일 경로
log_file = 'log.txt'

# 초기 로그 파일 생성
with open(log_file, 'w') as f:
    f.write("Training and Testing Results\n")
    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, epoch)
    test_loss, accuracy = evaluate(model, test_dataloader, criterion, epoch)

    epoch_result = {'epoch': epoch + 1, 'train_loss': train_loss, 'test_loss': test_loss, 'accuracy': accuracy}
    epoch_results.append(epoch_result)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print('-------------------')


# 대시보드 레이아웃
app.layout = html.Div([
    html.H1('Training and Testing Results'),
    dcc.Graph(
        id='loss-graph',
        figure={
            'data': [
                go.Scatter(
                    x=list(range(1, len(loss_data['train']) + 1)),
                    y=loss_data['train'],
                    mode='lines+markers',
                    name='Train Loss'
                ),
                go.Scatter(
                    x=list(range(1, len(loss_data['test']) + 1)),
                    y=loss_data['test'],
                    mode='lines+markers',
                    name='Test Loss'
                )
            ],
            'layout': go.Layout(
                title='Loss',
                xaxis={'title': 'Step'},
                yaxis={'title': 'Loss'}
            )
        }
    ),
    dcc.Graph(
        id='accuracy-graph',
        figure={
            'data': [
                go.Scatter(
                    x=list(range(1, len(accuracy_data) + 1)),
                    y=accuracy_data,
                    mode='lines+markers',
                    name='Accuracy'
                )
            ],
            'layout': go.Layout(
                title='Accuracy',
                xaxis={'title': 'Step'},
                yaxis={'title': 'Accuracy'}
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
