import numpy as np
from tqdm import tqdm

def initial(params):

    # Create initial state of the problem

    #OUTPUTS:
    #   u = temperature array with first column initialized and all other columns 0
    u = np.zeros((params['N'],params['M'])) # Temperature
    # Set initial conditions
    if params['constant_rad_heating'] == True: # if radiogenic heating is constant
        u[:,0] = -params['Hrad']/params['kc'] * (params['z']**2/2) + ((params['Hrad']/params['kc'])*params['L_crust'] + params['qm']/params['kc'])*params['z'] + params['Tsurf'] # initial crust temperature 
    elif params['constant_rad_heating'] == False: # if radiogenic heating is exponential
        for i in range(0,params['N']): #steady state geotherm
            c1 = params['qm']/params['kc'] - params['Hrad0']*params['hr']/params['kc']*np.exp(-params['L_crust']/params['hr']) # constant of integration
            c0 = params['Tsurf'] + params['Hrad0']*params['hr']**2/params['kc'] # constant of integration
            u[i,0] = -params['Hrad0']*params['hr']**2/params['kc'] * np.exp(-params['z'][i]/params['hr']) + c1*params['z'][i] + c0 # initial crust temperature  

    return u

def dT_dt(T,index,qi,params):

    # Use the method of lines to calculate the derivative (dT/dt) by calculating fluxes in and out of each spatial node
    #
    # INPUTS:
    # T: Temperature at current time step
    # index: index at which this timestep is happening in the integration function
    # qi: heat flux out due to ice sheet resistivity
    # params: dictionary of all constants used in calculation
    # 
    # OUTPUTS:
    # dT: the change in temperature from this timestep to the next timestep  

    dT = np.empty(np.shape(T))
    # q on staggered grid order (delta_x)^4 accurate
    q_s = np.zeros(len(T)-1)
    q_s[1:-1] = -params['kc']*(-0.5*(T[3:]) + 7.5*T[2:-1] - 7.5*T[1:-2] + 0.5*(T[:-3]))/(6*params['SPACE_STEP']) # fourth order centered difference
    q_s[0] = -params['kc']*(T[1]-T[0])/(params['SPACE_STEP']) # second order centered difference for i = 1/2
    q_s[-1] = -params['kc']*(T[-1]-T[-2])/(params['SPACE_STEP']) # second order centered difference for i = M - 1/2
   
    if params['ice'] == True:  # if there is an ice sheet
        dT[0] = (-q_s[0] + qi)/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']) # temperature change at top boundary

        dT[1:-1] = -(q_s[1::] - q_s[0:-1])/(params['rho_cp'] * params['SPACE_STEP']) + params['Hrad'][1:-1]/params['rho_cp'] # temperature change at central nodes

        dT[-1] = (params['delta_qm'][index] + q_s[-1])/(params['rho_cp'] * params['SPACE_STEP']) + params['Hrad'][-1]/params['rho_cp'] # temperature change at bottom boundary

        
        if T[0]+ params['TIME_STEP']*dT[0]  > params['Tmelt']: # if T at the top boundary exceeds melting point of ice then remove excess heat due to melting
            dT[0] = (-q_s[0] + qi - params['rho_cp']*params['SPACE_STEP']*(T[0]  - 273)/params['TIME_STEP'])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp'])
    elif params['ice'] == False: # if there is not an ice sheet
        dT[0] = 0
        dT[1:-1] = -(q_s[1::] - q_s[0:-1])/(params['rho_cp'] * params['SPACE_STEP']) + params['Hrad'][1:-1]/params['rho_cp']
        dT[-1] = (params['delta_qm'][index] + q_s[-1])/(params['rho_cp'] * params['SPACE_STEP']) + params['Hrad'][-1]/params['rho_cp']
    
    return dT

def integration_DF(t,T,params):

    # Integrate T through time using the Dufort-Frankel (DF) Method. The first timestep and the top and bottom of each iteration use Forward Time Centered Space (FTCS) to step forward. 
    # This is because DF requires the previous timestep to calculate the next time step and this is no possible to start. FTCS is useful at the top and bottom of the domain to 
    # both because DF uses the spatial step above and below (which isn't possible at the top and bottom of the domain) and this problem applies fixed fluxes at the boundaries and 
    # FTCS makes that much easier.
    # 
    # INPUTS:
    # t: time array with length M where M is the number of time steps
    # T: Temperature with shape (N,M) where N is the number of spatial steps 
    # params: dictionary with all constant used in the problem
    # 
    # OUTPUTS:
    # T: Temperature array now filled with shape (N,M)
    # Hmagma: excess heat from melting of the base of the crust and spread throughout the crust (N,M)
    # Un: Unloading number array with size (N,M)
    # visc: Viscosity array with size (N,M)
    # Qvol: magma volume generated to create Hmagma (M,1)
    # Qvol_total: magma volume calculated from delta_qm which would create the equivalent amount of heating(M,1)

    # constants used for the DF calculations
    alpha = 2*params['kc']*params['TIME_STEP']/(params['rho_cp']*params['SPACE_STEP']**2)
    delta = (1-alpha)/(1+alpha)
    chi = alpha/(1+alpha)
    xeta = 2*params['TIME_STEP']*params['Hrad'][1:-1] / ((1+alpha)*params['rho_cp'])



    # Flux through the ice sheet which will be set as the top boundary condition. Only first flux is calculated because qi is dependent on the temperature
    qi = np.zeros(len(t))
    qi[0] = -(params['ki']/params['L_ice'][0])*(T[0,0]- params['Tsurf'])

    # First time step using FTCS
    T[:,1] = T[:,0] + params['TIME_STEP']*dT_dt(T[:,0],0,qi[0],params)

    # create array to store melting fluxes which are calculated at each timestep
    qmelt = np.zeros(len(t))
    Hmagma = np.zeros(np.shape(T))

    # create arrays to store magma volumes for excess melting and total volume 
    Qvol = np.zeros(len(t)) # excess melting volume
    Qvol_total = np.zeros(len(t)) # total magma volume rate for all delta_qm
    N = 6 # Number of dykes required to achieve correct amount of melting. Used for Qvol_total calculations.
    
    # only used for the Collapse experiment
    # collapse_start = np.arange(2.8,5,0.0337) # set the time of the start of each collapse and the time between the start of each collapse

   

    for ii in tqdm(np.arange(1,params['M'])):

        if ii+1 >= params['M']: # dont execute past last time step
             break
        
        # calculate total volume rate for current delta_qm value 
        Qvol_total[ii] = params['delta_qm'][ii]*params['L_crust']*10*N / (params['efficiency']*params['rho_crust'] * (params['cp_crust']*np.mean(params['Tliq'] - T[np.argwhere(params['z'] == 5000)[0][0]:-1,ii]) + params['Lf'])) # rate of magma injection into the crust calculated using delta_qm
        
        # calculate flux through the ice sheet at each timestep
        qi[ii] = (params['ki']/params['L_ice'][ii])*(T[0,ii]- params['Tsurf']) # insulated heat flux through the ice sheet

        # calculate the next temperature at the top of the spatial domain using FTCS
        if params['ice'] == True:
            Tguess = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp'])) # guess a temperature for the next time step
            
            # Check to see if the next temperature will be above 273 K. If yes, add in the melt flux to this timestep and recalculate next temperature. If no, set next temperature to Tguess
            if Tguess > params['Tmelt']:
                qmelt[ii] = params['rho_cp']*params['SPACE_STEP']*(Tguess  - params['Tmelt'])/params['TIME_STEP']
                T[0,ii+1] = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii] - qmelt[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']))
            else:
                T[0,ii+1] = Tguess
        elif params['ice'] == False:
            T[0,ii+1] = params['Tsurf']

        ##################################################################################################################################
        ############### This part is for plotting experiment 1 and Collapse Experiment and I will delete it once the paper/thesis is done just in case I need to replot #########################

        # if params['ice'] == True:

            ########### Start Experiment 1 ###########
            # if t[ii] < 2.5*params['tau_d']:
            #         T[0,ii+1] = params['Tsurf']

            # else:
            #     Tguess = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']))
                
            #     # Check to see if the next temperature will be able 273 K. If yes, add in the melt flux to this timestep and recalculate next temperature. If no, set next temperature to Tguess
            #     if Tguess > params['Tmelt']:
            #         qmelt[ii] = params['rho_cp']*params['SPACE_STEP']*(Tguess  - params['Tmelt'])/params['TIME_STEP']
            #         T[0,ii+1] = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii] - qmelt[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']))
            #     else:
            #         T[0,ii+1] = Tguess
            ########### End Experiment 1 ###########

            ########### Start Collapse Experiment ###########
            # collapse = np.zeros(len(collapse_start))
            # for kk,tt in enumerate(collapse_start):
            #     collapse[kk] = np.all((t[ii] > tt*params['tau_d'],t[ii] < (tt+0.017)*params['tau_d']))
            
            # if t[ii] < 2.5*params['tau_d']:
            #     T[0,ii+1] = params['Tsurf']
                

            # elif np.any(collapse):
            #     T[0,ii+1] = params['Tsurf']
            #     # continue

            # else:
            #     Tguess = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']))
            
            #     # Check to see if the next temperature will be able 273 K. If yes, add in the melt flux to this timestep and recalculate next temperature. If no, set next temperature to Tguess
            #     if Tguess > params['Tmelt']:
            #         qmelt[ii] = params['rho_cp']*params['SPACE_STEP']*(Tguess  - params['Tmelt'])/params['TIME_STEP']
            #         T[0,ii+1] = T[0,ii] + params['TIME_STEP']*((params['kc']*(T[1,ii]-T[0,ii])/params['SPACE_STEP'] - qi[ii] - qmelt[ii])/(params['rho_cp']*params['SPACE_STEP']) + params['Hrad'][0]/(params['rho_cp']))
            #     else:
            #         T[0,ii+1] = Tguess
            ########### End Collapse Experiment ###########


            
        # elif params['ice'] == False:
        #     T[0,ii+1] = params['Tsurf']
        ##################################################################################################################################
            
        # calculate next tempreature at the bottom of the spatial domain (crust/mantle boundary) using FTCS
        Tguess2 = T[-1,ii] + params['TIME_STEP']*((params['delta_qm'][ii] - params['kc']*(T[-1,ii]-T[-2,ii])/params['SPACE_STEP'])/(params['rho_cp']*params['SPACE_STEP'])+ params['Hrad'][-1]/(params['rho_cp']))
        
        # Check to see if the next temperature will be above liquidus temp. If yes, add in the melt flux to this timestep and recalculate temp. If no, set next temperature to Tguess2
        if Tguess2 > params['Tliq']:
            qmelt_magma = params['rho_cp']*params['SPACE_STEP']*(Tguess2  - params['Tliq'])/params['TIME_STEP']
            T[-1,ii+1] = T[-1,ii] + params['TIME_STEP']*((params['delta_qm'][ii] - params['kc']*(T[-1,ii]-T[-2,ii])/params['SPACE_STEP']-qmelt_magma)/(params['rho_cp']*params['SPACE_STEP'])+ params['Hrad'][-1]/(params['rho_cp']))
            Qvol[ii] = qmelt_magma/(params['rho_crust']*params['Lf'])
            Hmagma[:,ii] = params['efficiency'] * params['rho_crust']*(params['cp_crust']*(params['Tliq']-T[:,ii])+params['Lf'])*Qvol[ii]*params['dz']/(params['dyke_length'])
            xeta =  2*params['TIME_STEP']*(params['Hrad'][1:-1]+Hmagma[1:-1,ii]) / ((1+alpha)*params['rho_cp'])
            Tguess3 = delta*T[1:-1,ii-1] + chi*(T[2::,ii] + T[0:-2,ii]) + xeta

            # Check to see if additional nodes above the bottom boundary exceed Tliq. If yes remove excess heat flux.
            if np.any(Tguess3>990+273):# params['Tliq']):
                indices_above_liquidus = np.argwhere(Tguess3>params['Tliq'])
                for kk in np.flip(indices_above_liquidus):
                    Hmagma[1:-1,ii][kk[0]] = 0
                xeta =  2*params['TIME_STEP']*(params['Hrad'][1:-1]+Hmagma[1:-1,ii]) / ((1+alpha)*params['rho_cp'])
                T[1:-1,ii+1] = delta*T[1:-1,ii-1] + chi*(T[2::,ii] + T[0:-2,ii]) + xeta
            else:
                T[1:-1,ii+1] = Tguess3
        else:
            T[-1,ii+1] = Tguess2
            xeta = 2*params['TIME_STEP']*params['Hrad'][1:-1] / ((1+alpha)*params['rho_cp'])

            # calculate next temperatures within the spatial domain using DF
            T[1:-1,ii+1] = delta*T[1:-1,ii-1] + chi*(T[2::,ii] + T[0:-2,ii]) + xeta
        
        
        

    # # calculate viscosity using Arrhenius Viscosity Law
    visc = params['A']*np.exp(params['G']/(params['Bmu']*T)) # Viscosity


    # #calculate Unloading number
    Un = (visc/(params['year']*params['Eeff']))/params['tf'] # Unloading Number

    return T,Hmagma,Un,visc,Qvol,Qvol_total
