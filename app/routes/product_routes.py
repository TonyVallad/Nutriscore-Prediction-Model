from flask import Blueprint, render_template, redirect, url_for, current_app, request
from config import Config
import math

product_bp = Blueprint('product_bp', __name__)

# Training Data Route
@product_bp.route('/training_data')
def training_data():
    """
    Displays the training data in a paginated format.

    If the products DataFrame has not been loaded, redirects to the 'Loading dataframe' template.

    Otherwise, retrieves the products DataFrame from the app config, paginates it, and
    extracts unique Nutriscore grades and categories. Passes these to the 'training_data.html'
    template to render the page.

    Returns:
        HTML: The 'training_data.html' template.
    """

    # Checks if the dataframe is already loaded
    if 'PRODUCTS_DF' not in current_app.config:
        # Send to a 'Loading dataframe' template
        return redirect(url_for('main.loading_data'))

    # Retrieve the products DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']

    # Pagination parameters
    page = request.args.get('page', 1, type=int)  # Get the current page, default is 1
    per_page = 50  # Number of products to show per page

    # Calculate total pages
    total_products = len(products)
    total_pages = math.ceil(total_products / per_page)

    # Paginate the DataFrame
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_products = products.iloc[start_index:end_index].to_dict(orient='records')

    # Extract unique Nutriscore grades, sorted alphabetically
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())  # Extract unique values and sort alphabetically

    # Retrieve all unique Categories for the sidebar
    cat_list = sorted(products['pnns_groups_1'].dropna().unique())  # Extract unique values and sort alphabetically

    return render_template('training_data.html',
                           nutriscore_grades=nutriscore_grades,
                           cat_list=cat_list,
                           products=paginated_products, 
                           page=page, 
                           total_pages=total_pages,
                           total_products=total_products)

# Search Route
@product_bp.route('/search', methods=['GET', 'POST'])
def search():
    """
    Displays the search page with a search form and a sidebar containing
    unique Categories and Nutriscore grades.

    Retrieves the products DataFrame from the app config, extracts unique
    Nutriscore grades and categories, and passes them to the 'search.html'
    template to render the page.

    Returns:
        HTML: The 'search.html' template.
    """

    # Loads the products DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']
    
    # Extract unique Nutriscore grades, sorted alphabetically
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())

    # Retrieve all unique Categories for the sidebar
    cat_list = sorted(products['pnns_groups_1'].dropna().unique())

    return render_template('search.html',
                           nutriscore_grades=nutriscore_grades,
                           cat_list=cat_list)

# Search Results Route
@product_bp.route('/search_results', methods=['GET'])
def search_results():
    """
    Handles the search results for products based on user-submitted search criteria.

    Retrieves the products DataFrame from the app config and checks if the search form
    was explicitly submitted. If submitted, filters the products based on the search
    parameters such as search term, selected columns, Nutriscore grades, and categories.
    The filtered results are saved to the app config for future reference.

    If no new search is performed, uses the existing search results stored in the app config.
    Also retrieves unique Nutriscore grades and categories for sidebar display.

    The results are paginated and rendered in the 'search_results.html' template.

    Returns:
        HTML: The 'search_results.html' template with paginated search results,
              category list, Nutriscore grades, and pagination metadata.
    """

    # Get the DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']

    # Check if the form was explicitly submitted
    form_submitted = request.args.get('submitted', '') == 'true'

    # Handle form submission
    if form_submitted:
        # Perform a new search based on the form parameters
        search_results = products

        # Retrieve search parameters from GET request
        search_term = request.args.get('search_term', '').strip().lower()
        search_columns = request.args.getlist('search_columns')
        selected_grades = request.args.getlist('nutriscore_grades')
        pnns_groups_1 = request.args.getlist('pnns_groups_1')

        # Apply filters based on the retrieved parameters
        if selected_grades:
            search_results = search_results[search_results['nutriscore_grade'].isin(selected_grades)]
        
        if pnns_groups_1:
            search_results = search_results[search_results['pnns_groups_1'].isin(pnns_groups_1)]

        if search_term and search_columns:
            search_columns = [col for col in search_columns if col in search_results.columns]
            search_results = search_results[
                search_results[search_columns]
                .apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
            ]

        # Save the filtered results to the app config
        current_app.config['SEARCH_RESULTS_DF'] = search_results
    else:
        # Use the existing search results if no new search was performed
        search_results = current_app.config.get('SEARCH_RESULTS_DF', products)

    # Retrieve all unique Nutriscore grades for the sidebar
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())

    # Retrieve all unique Categories for the sidebar
    cat_list = sorted(products['pnns_groups_1'].dropna().unique())

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 50
    total_products = len(search_results)
    total_pages = math.ceil(total_products / per_page)

    # Paginate the DataFrame
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_products = search_results.iloc[start_index:end_index].to_dict(orient='records')

    return render_template('search_results.html', 
                           products=paginated_products,
                           cat_list=cat_list,
                           page=page, 
                           total_pages=total_pages,
                           nutriscore_grades=nutriscore_grades,
                           total_products=total_products)