from django import forms

class AddictionForm(forms.Form):
    daily_screen_time = forms.FloatField(label="Daily Screen Time (Hours)")
    social_media = forms.FloatField(label="Social Media (Hours)")
    gaming = forms.FloatField(label="Gaming (Hours)")
    sleep_hours = forms.FloatField(label="Sleep (Hours)")
    notifications = forms.IntegerField(label="Daily Notifications")
    app_opens = forms.IntegerField(label="Daily App Opens")
    weekend_usage = forms.FloatField(
        label="Total Weekend Usage", 
        help_text="Enter total hours for Saturday and Sunday combined (e.g., 12.5)"
    )
    