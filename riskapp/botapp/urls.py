from botapp import views
from django.urls import path

app_name = "botapp"
urlpatterns = [
    path("", views.gradio_chatbot_view, name="gradio_chatbot_view"),
]
