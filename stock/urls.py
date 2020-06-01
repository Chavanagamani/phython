from django.urls import path

from . import views

urlpatterns = [
    path('', views.user_details, name='user_details'),
    path('user/<int:user_id>/', views.user_get_by_id, name='user_get_by_id'),
    path('startModel', views.create_model, name='create_model'),
    path('getStockPrediction/<int:data>', views.run_model, name='run_model')
]
