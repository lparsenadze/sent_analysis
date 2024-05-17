from flask_swagger_ui import get_swaggerui_blueprint

flask_swagger_ui = get_swaggerui_blueprint(
    "/swagger",
    '/static/swagger.json',
    config={
        'ml-preparation': 'Access API'
    }
)