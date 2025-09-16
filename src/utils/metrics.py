import torch
from sklearn.metrics import classification_report

def evaluate_model(model, data_loader, device, class_names):
    """
    تقييم النموذج على DataLoader معين وإرجاع:
    - accuracy: دقة النموذج
    - report: تقرير يحتوي Precision, Recall, F1 لكل فئة
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # حساب الدقة
    accuracy = (torch.tensor(y_true) == torch.tensor(y_pred)).sum().item() / len(y_true)

    # تقرير كامل
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    return accuracy, report
