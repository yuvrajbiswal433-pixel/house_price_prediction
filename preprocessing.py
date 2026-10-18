
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# ============================================================
# 1. COLUMNS TO DROP
# ============================================================

cols2 = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType',
    'FireplaceQu', 'LotFrontage', 'Id', 'PoolArea',
    '3SsnPorch', 'MiscVal', 'LowQualFinSF'
]


class dropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=cols2, errors='ignore')


# ============================================================
# 2. NUMERICAL IMPUTATION
# ============================================================

cols1 = [
    'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'GarageCars',
    'GarageArea'
]


class numimputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = (
            list(self.columns) if self.columns is not None else cols1
        )

        self.imputer_ = SimpleImputer(strategy='median')
        self.imputer_.fit(X[self.columns_])

        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns_] = self.imputer_.transform(X[self.columns_])
        return X


# ============================================================
# 3. ORDINAL CATEGORY ORDERS
# ============================================================

bsmt_qual_order = ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
bsmt_cond_order = ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

bsmt_exposure_order = ['Missing', 'No', 'Mn', 'Av', 'Gd']

bsmtfintype_order = [
    'Missing', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'
]

garage_finish_order = ['Missing', 'Unf', 'RFn', 'Fin']

garage_qual_cond_order = [
    'Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'
]

exter_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

heatingqc_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

kitchenqual_order = ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']


# ============================================================
# 4. ORDINAL ENCODER
# ============================================================

cols3 = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'GarageFinish',
    'GarageQual', 'GarageCond'
]


class ordinalencoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        categories = [
            exter_order,
            exter_order,
            bsmt_qual_order,
            bsmt_cond_order,
            bsmt_exposure_order,
            bsmtfintype_order,
            bsmtfintype_order,
            heatingqc_order,
            kitchenqual_order,
            garage_finish_order,
            garage_qual_cond_order,
            garage_qual_cond_order
        ]

        self.encoder_ = OrdinalEncoder(
            categories=categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

        self.encoder_.fit(X[cols3])
        return self

    def transform(self, X):
        X = X.copy()
        X[cols3] = self.encoder_.transform(X[cols3])
        return X


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

ohe_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation',
    'Heating', 'CentralAir', 'Electrical', 'Functional',
    'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition'
]


class OHEEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )

        self.encoder_.fit(X[ohe_cols])
        return self

    def transform(self, X):
        X = X.copy()

        encoded = self.encoder_.transform(X[ohe_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder_.get_feature_names_out(ohe_cols),
            index=X.index
        )

        X = X.drop(columns=ohe_cols)
        X = pd.concat([X, encoded_df], axis=1)

        return X


# ============================================================
# 6. RARE CATEGORY GROUPING
# ============================================================

class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.rules = {
            'Condition1': 10,
            'Condition2': 10,
            'RoofMatl': 5,
            'Exterior1st': 5,
            'Exterior2nd': 5,
            'SaleType': 5,
            'Neighborhood': 2
        }

    def fit(self, X, y=None):
        self.rare_categories_ = {}

        for col, threshold in self.rules.items():
            counts = X[col].value_counts(dropna=True)

            self.rare_categories_[col] = counts[
                counts < threshold
            ].index.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        for col, rare_values in self.rare_categories_.items():
            mask = X[col].isin(rare_values)
            X.loc[mask, col] = 'Other'

        return X


# ============================================================
# 7. MISSING CATEGORICAL VALUE IMPUTATION
# ============================================================

cols4 = [
    'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
    'BsmtCond', 'BsmtQual', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'KitchenQual'
]


class MissingImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[cols4] = X[cols4].fillna('Missing')
        return X


# ============================================================
# 8. LOG TRANSFORMATION
# ============================================================

cols5 = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    'ScreenPorch', 'MoSold', 'YrSold'
]


class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        values = X[cols5]

        if (values <= -1).any().any():
            raise ValueError(
                'LogTransformer requires values greater than -1.'
            )

        X[cols5] = np.log1p(values)

        return X


# ============================================================
# 9. REMAINING CATEGORICAL IMPUTATION
# ============================================================

cat_cols = [
    'MSZoning', 'Utilities', 'Exterior1st',
    'Exterior2nd', 'Functional', 'SaleType'
]


class CatImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy='most_frequent')
        self.imputer_.fit(X[cat_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[cat_cols] = self.imputer_.transform(X[cat_cols])
        return X