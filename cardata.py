# helper functions for working with the car data 
import numpy as np 
import pandas as pd 
#from numba import njit # for speed

# labels for some of the variables in the Verboven car dataset
market_name        = ['Belgium','France','Germany','Italy','UK']; 
class_name         = ['subcompact','compact','intermediate','standard','luxury']; 
origin_code_name   = ['France','Germany','Italy','JapanKorea','Spain','Sweden','UK','EasternEurope','US','Brazil']; 
location_code_name = ['Belgium','CheckRepublic','France','Germany','Italy','Japan','Korea','Netherlands','Romania','Spain','Sweden','UK','Russia','Yugoslavia','Poland','US','Finland','Australia','Hungary','Portugal','India','Mexico']; 
firm_code_name     = {1: 'AlfaRomeo', 2: 'BMW', 3: 'Toyota', 4: 'Fiat', 5: 'Ford', 6: 'Honda', 7: 'Hyundai', 8: 'DeTomaso', 9: 'Kia', 10: 'Lada', 11: 'Mazda', 12: 'Mercedes', 13: 'Mitsubishi', 14: 'Nissan', 15: 'GeneralMotors', 16: 'Peugeot', 17: 'Porsche', 18: 'Renault', 19: 'Rover', 20: 'Saab', 21: 'Seat', 22: 'AZNP', 23: 'FujiHI', 24: 'Suzuki', 25: 'Toyota', 26: 'VW', 27: 'Volvo', 28: 'Yugo', 29: 'Daewoo', 30: 'Daimler', 31: 'DAF', 32: 'Jaguar', 33: 'TalbotSimcaHillmanSunbeam', 34: 'TalbotMatra', 42: 'Lancia'}

def dataframe2matrix(cars, var, J, T, N): 
    """dataframe2matrix: take a variable in cars and convert to a 
        matrix. 
        
        INPUTS: 
            cars: pandas dataframe 
            var: string, the name of the variable requested
            J,T,N: integers, denoting dimensions of
                J: choiceset (variable 'co' in index)
                T: time (variable 'ye' in index)
                N: market (variable 'ma' in index)
            
        OUTPUT: 
            mat: J*T*N matrix with the requested variable
    """
    
    # check the index
    assert isinstance(cars.index,pd.MultiIndex), 'index should be (ma, ye, co) multiindex'
    assert cars.index.names == ['ma','ye','co'], f'Dataframe, cars, must have index (ma, ye, co)'
    assert cars.index.is_lexsorted(), 'index should be sorted'
    
    countries = cars.index.get_level_values('ma').unique().values
    assert len(countries) == N 
    
    # unstacking gives us the different cars ('co') as columns in sorted order
    cc = cars[var].unstack(level='co')
    
    # figure out the datatype of the requested variable 
    # (exception for objects like categoricals which numpy does not understand)
    if cars[var].dtype in [float, int]: 
        this_dtype = float
    else: 
        this_dtype = object 
        
    # initialize output 
    mat = np.empty((J,T,N), dtype=this_dtype)

    for ic, c in enumerate(countries): 
        if c in cc.index.get_level_values('ma'): 
            mat_ = cc.loc[c].values # this is T*J, but we need J*T
            mat[:,:,ic] = mat_.T   # ... so we transpose it 
            
    return mat 


def compute_xi_from_estimates(res,cars,fixed_effects=['co','ma','ye'],DOPRINT=False): 
    '''
        INPUTS: 
            res: (statsmodels linear regression results) 
            cars: pandas df 
            fixed_effects: coefficients to unpack. Only certain ones allowed (checked)

        OUTPUT: 
            xi: J*T*N matrix of "fixed effects" (utility shifters for each car)
    '''
    assert 'ma' in cars.columns, f'var "ma" not in columns of cars: did you index it? '
    
    allowed_fixed_effects = ['co','ma','ye','frm']
    for v in fixed_effects: 
        assert v in allowed_fixed_effects, f'fixed effect "{v}" not implemented'

    countries = np.sort(cars['ma'].unique())
    years     = np.sort(cars['ye'].unique())
    carcodes  = np.sort(cars['co'].unique())

    T = len(years)
    N = len(countries)
    J = len(carcodes)

    xi = np.zeros((J,T,N)) + res.params.loc['Intercept']

    if 'frm' in fixed_effects: 
        c = cars.set_index(['ma','ye','co']).sort_index()
        frms = dataframe2matrix(c, 'frm', J, T, N)
        for f in cars.frm.unique(): 
            nam = f'C(frm)[T.{f}]'
            if nam in res.params.index: 
                I = frms == f 
                xi[I] += res.params.loc[nam]

    if 'co' in fixed_effects: 
        co_not_found = [] 
        for j in range(J): 
            co = carcodes[j]
            nam = f'C(co)[T.{j}]'
            if nam in res.params.index: 
                # add coefficient value to all cars of this type 
                xi[j,:,:] += res.params.loc[nam]
            else: 
                co_not_found.append(co)

        if (len(co_not_found) > 0) & DOPRINT: 
            print(f'{len(co_not_found)} cars not found in estimates')

    if 'ye' in fixed_effects: 
        for t in range(T): 
            ye = years[t]
            nam = f'C(ye)[T.{ye}]'
            if nam in res.params.index: 
                # add year effect to all observations in that year 
                xi[:,t,:] += res.params.loc[nam]
            else: 
                if DOPRINT: 
                    print(f'Year {ye} not in coef')

    if 'ma' in fixed_effects: 
        for i in range(N): 
            ma = countries[i]
            nam = f'C(ma)[T.{ma}]'
            if nam in res.params.index: 
                # add country fixed effect to all observations in this country 
                xi[:,:,i] += res.params.loc[nam]
            else: 
                if DOPRINT: 
                    print(f'Market {ma} not in coef')

    return xi 

# @njit
def utils(xi, z, beta, available):
    """
        INPUTS 
            xi: J*T*N matrix of fixed effects
            z: Nz*J*T*N matrix of car characteristics 
            beta: Nz vector (marginal utility of each car characteristic)
            available: J*T*N matrix af booleans (True if car is available for purchase)

        OUTPUT: 
            u: J*T*N matrix of utilities 
    """ 
    J, T, N = available.shape
    u = np.zeros((J,T,N))
    
    u[available] += xi[available]
    
    for i in range(N):
        for t in range(T): 
            for j in range(J):
                if available[j,t,i]:
                    u[j,t,i] = np.dot(z[:,j,t,i], beta) 
                    
    u[available == False] = -np.inf

    return u

def choice_prob(xi, z, beta, available, rho=None, nest_id=None): 
    """
        INPUTS: 
            xi: J*N*T matrix of fixed effects
            z: J*Nz matrix of car characteristics
            beta: Nz vector (marginal utility of each car characteristic)
            available: J*T*N matrix of booleans
            nest_ids: J*T*N matrix of base 0 integers for nest 
            rho: scalar parameter

        OUTPUT: 
            P: J*T*N matrix of choice probabilities 
    """

    J,T,N = available.shape 
    Nz = z.shape[0]
    assert len(beta) == Nz
    
    u = utils(xi, z, beta, available) 

    if rho == None: 
        P = compute_ccps_inner(u, available)
    else: 
        # quick check 
        # Construct inclusive values 
        unique_nest_ids = np.unique(nest_id[available])
        L = len(unique_nest_ids)
        assert (min(unique_nest_ids) == 0) and (max(unique_nest_ids) == L-1), f'nesting_id should be base 0'
        P = compute_ccps_nested(u, available, rho, nest_id, unique_nest_ids)
    
    return P

#@njit
def max_dim_0(u): 
    '''max_dim_0: Take the maximum along the first dimension (i.e.i) of three. 
        Fully equivalent to u.max(0)

        INPUT: 
            u: 3-dimensional numpy array 
        OUTPUT: 
            umax: (2-dimensional numpy array) umax = u.max(0)
    '''
    J,T,N = u.shape 
    umax = -np.inf * np.ones((T,N))
    for t in range(T): 
        for i in range(N): 
            for j in range(J): 
                umax[t,i] = max(umax[t,i], u[j,t,i])
    
    return umax

#@njit
def compute_ccps_inner(U, available): 
    '''compute_ccps_inner: seperated out function so it can be jitted 
        INPUT: 
            U: J*T*N matrix of utilities 
            available: J*T*N matrix of booleans (True if car j is available in year t in market i)
        OUTPUT: 
            P: J*T*N matrix of choice probabilities 
    '''
    J,T,N = available.shape 
    
    # max rescale: equivalent to umax = U.max(0), but np.amax() is not supported by numba 
    umax = max_dim_0(U) # T*N
    u = U - umax # J*T*N - T*N = J*T*N
    
    # Construct denominator 
    denom = np.zeros((T,N))
    for i in range(N):
        for t in range(T): 
            for j in range(J):
                if available[j,t,i]:
                    denom[t,i] += np.exp(u[j,t,i])

    # outside option 
    denom += np.exp(0.0 - umax)
                    
    # computer probabilities 
    P = np.zeros((J,T,N))
    for i in range(N):
        for t in range(T): 
            for j in range(J):
                if available[j,t,i]:
                    P[j,t,i] = np.exp(u[j,t,i]) / denom[t,i]
                    
    return P

#@njit
def compute_ccps_nested(U, available, rho, nest_id, unique_nest_ids): 
    '''compute_ccps_inner: seperated out function so it can be jitted 
        OUTPUT: 
            P: J*T*N matrix of choice probabilities 
    '''
    J,T,N = available.shape 
    L = len(unique_nest_ids)
    
    # max rescale: equivalent to umax = U.max(0), but np.amax() is not supported by numba 
    umax = max_dim_0(U)
    u = U - umax

    # all utilities are divided by rho
    u = u/rho
    
    denom = np.zeros((T,N,L))
    for i in range(N):
        for t in range(T): 
            for j in range(J):
                if available[j,t,i]:
                    l = nest_id[j,t,i]
                    denom[t,i,l] += np.exp(u[j,t,i])

    # sum over nests 
    sum_denom = np.sum(denom ** rho, 2)

    # outside option: its own nest 
    sum_denom += (np.exp(0.0 - umax)) ** rho

    # exponentiate inclusive values 
    for l in range(L): 
        denom[:,:,l] = denom[:,:,l] ** (rho-1.0)
                    
    # computer probabilities 
    P = np.zeros((J,T,N))
    for i in range(N):
        for t in range(T): 
            for j in range(J):
                if available[j,t,i]:
                    l = nest_id[j,t,i]
                    P[j,t,i] = np.exp(u[j,t,i]) * denom[t,i,l] / sum_denom[t,i]
                    
    return P

def ls_crit(beta, xi, z, available, s): 
    '''ls_crit: least squares criterion for estimation using market shares 
        INPUTS: 
            beta: Nz-vector, parameters (marginal utility for each car characteristic)
            xi: J*T*N matrix of car fixed effects 
            z: Nz*J*T*N matrix of car characteristics 
            available: J*T*N matrix of booleans (=True if car j can be bought in year t in market i)
            s: J*T*N matrix of market shares (data)

        OUTPUT: 
            L2: scalar, sum of squared residuals

        EXAMPLE: 
            x0 = np.array([-0.83701734, -0.35462293,  2.66424945, -0.0141676 ])
            res = minimize(cardata.ls_crit, x0, args=(xi,z,available,s,rho,nest_id),
                   method='nelder-mead', options={'maxiter':1000})

    '''
    probs = choice_prob(xi,z,beta,available)
    d = probs[available] - s[available]
    L2 = (d ** 2).mean()
    return L2


def assert_matrix_consistencies(cars_in, s, z, zvars): 
    '''assert_matrix_consistencies: checks that reshaping went well
        INPUTS: 
            cars_in: pandas dataframe
            s: J*T*N matrix of market shares for cars 
            z: Nz*J*T*N matrix of car characteristics 
            zvars: Nz-list of strings: names of the characteristics
        OUTPUT: 
            boolean (or throws an error if it fails)
    '''
    if not 'ma' in cars.columns: 
        cars = cars_in.reset_index()
    else: 
        cars = cars_in

    countries = np.sort(cars['ma'].unique())
    years     = np.sort(cars['ye'].unique())
    carcodes  = np.sort(cars['co'].unique())

    T = len(years)
    N = len(countries)
    J = len(carcodes)


    for ic in range(N): # loop over countries 
        for t in range(T): # ... and years 
            # find data for country number ic and year t
            cc = cars.loc[countries[ic]].loc[years[t]]

            #find cars in that country and that year 
            cars__ = cc.index.get_level_values('co')
            for j in range(J):
                if carcodes[j] in cars__: 
                    # check the market shares 
                    assert cc.loc[carcodes[j], 's'] == s[j,t,ic]

                    # check each of the car characteristics (in zvars)
                    for iz in range(len(zvars)): 
                        zvar = zvars[iz] # read out the name of the variable
                        assert cc.loc[carcodes[j]][zvar] == z[iz,j,t,ic]

    print('success! verified that z and s matrices are correctly created ')

    return True

def process_data(cars, xvars=['eurpr', 'we', 'hp', 'li']): 

    cars.sort_values(['ma', 'ye', 'co'], inplace=True)

    cars.we = cars.we / 1000 # rescale from kg to tonnes: more sensible coefs. 

    cars['market'] = cars['ma'].map({i+1:s for i,s in enumerate(market_name)}).astype('category')
    cars['car_class'] = cars['cla'].map({i+1:s for i,s in enumerate(class_name)}).astype('category')
    cars['firm'] = cars['frm'].map(firm_code_name).astype('category')
    cars['firm_country'] = cars['org'].map({i+1:s for i,s in enumerate(origin_code_name)}).astype('category')
    cars['firm_production'] = cars['loc'].map({i+1:s for i,s in enumerate(location_code_name)}).astype('category')

    cars.sort_values(['ma', 'ye', 'qu'], ascending=False, inplace=True)

    # Size of the market: this can have important bearings on the estimates. 
    cars['pop'] = cars['pop']/10. # half are families, half are too young/too old, many do not consider a new car

    # sales in class: relevant 
    cars['sum_qu'] = cars.groupby(['ma', 'ye']).qu.transform('sum') # sold cars in this market-year 
    cars['sum_qu_class'] = cars.groupby(['ma', 'ye', 'cla']).qu.transform('sum') # sold cars in market-year-class 
    cars['num_in_class'] = cars.groupby(['ma', 'ye', 'cla']).qu.transform('count')
    cars['market_share_of_class'] = cars['sum_qu_class'] / cars['sum_qu'] # share of sales to this class out of all sales
    #cars['market_share_in_class'] = cars['qu'] / cars['sum_qu_class'] # share of sales in class that goes to this car 
    cars['market_share_inside'] = cars['qu'] / cars['sum_qu'] # share of sales going to this car 
    cars['market_share'] = cars['qu'] / cars['pop'] # share of the market buying this particular car 
    cars['market_share_outside'] = 1.0 - cars['sum_qu'] / cars['pop'] # share of the market not buying a car 
    cars['d_l_market_share'] = np.log(cars['market_share']) - np.log(cars['market_share_outside'])
    cars['d_l_market_share_of_class'] = np.log(cars['market_share_of_class']) - np.log(cars['market_share_outside'])

    # compute logarithms 
    zvars = [f'log_{x}' for x in xvars]
    Nz = len(zvars)
    for X in xvars: 
        cars[f'log_{X}'] = np.log(cars[X])

    cars['logpr'] = np.log(cars['eurpr'])
    cars['logqu'] = np.log(cars['qu'])

from statsmodels.formula.api import ols

def estimate(cars, yvar='d_l_market_share', zvars=['log_eurpr', 'log_we', 'log_hp', 'log_li'], fixed_effects=['frm', 'ye', 'ma'], NESTEDLOGIT=False): 
    print_these = [z_ for z_ in zvars] 

    model_spec = f'{yvar} ~ {zvars[0]}'
    for v in zvars[1:]: 
        assert v in cars.columns, f'Requested RHS var, "{v}" not in cars.columns'
        model_spec += f' + {v}'

    for fe in fixed_effects: 
        assert fe in cars.columns, f'Requested fixed effect, "{fe}" not in cars.columns (maybe in index?)'
        model_spec += f' + C({fe})'

    if NESTEDLOGIT: 
        # add the (log diff of the) market share of the class 
        model_spec += f' + d_l_market_share_of_class'
        print_these = print_these + ['d_l_market_share_of_class']

    m = ols(model_spec, data=cars.reset_index())

    res = m.fit()

    print(f'--- Parameter estimates (omitting fixed effects) ---')
    print(res.params.loc[print_these])

    # extract parameters needed
    betahat = res.params.loc[zvars]
    xi = compute_xi_from_estimates(res, cars, fixed_effects=['frm', 'ye', 'ma'])
    if NESTEDLOGIT: 
        rho = 1.0 - res.params.loc['d_l_market_share_of_class']
    else: 
        rho = None

    return betahat, xi, rho

