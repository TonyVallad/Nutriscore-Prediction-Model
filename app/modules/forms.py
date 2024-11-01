from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SubmitField, SelectField
from wtforms.validators import DataRequired, Optional, NumberRange

class NutriScoreForm(FlaskForm):
    product_name = StringField('Product Name', validators=[DataRequired()])
    
    # Main food group as a dropdown list
    pnns_groups_1 = SelectField('Main Food Group (pnns_groups_1)', choices=[], validators=[DataRequired()])

    # Nutritional Content Section (All relevant fields)
    energy_kcal_100g = DecimalField('Energy (kcal per 100g)', validators=[Optional(), NumberRange(min=0)])
    fat_100g = DecimalField('Fat (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    saturated_fat_100g = DecimalField('Saturated Fat (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    sugars_100g = DecimalField('Sugars (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    proteins_100g = DecimalField('Proteins (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    sodium_100g = DecimalField('Sodium (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    salt_100g = DecimalField('Salt (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    fiber_100g = DecimalField('Fiber (g per 100g)', validators=[Optional(), NumberRange(min=0)])
    fruits_vegetables_nuts_estimate_from_ingredients_100g = DecimalField('Fruits, Vegetables, Nuts Estimate (per 100g)', validators=[Optional(), NumberRange(min=0)])

    # Submit button
    submit = SubmitField('Predict')
