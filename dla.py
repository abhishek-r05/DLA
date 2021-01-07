# Import required libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors
import pylab as pl
import pickle

class dla():
    """
        fun
    """
    def __init__(self, demo=True, radius_n_k=[0, 0, 0]):
        """
            Demo       : if to quickly initialize for radius = 50, particle count = 2000, stickness = 1
            radius_n_k : list [radius,particle,stickness]
        """

        if demo:
            self.radius = 50
            self.max_count = 2000
            self.k = 1
        else:
            # No error checks on variable. Pass it wisely.
            self.radius, self.max_count, self.k = radius_n_k

        # Total area of grid  + 1 to make it odda nd symmetric around center.
        self.square_size = self.radius * 2 + 1

        # Create Matrix and fill with zeroes.
        matrix = np.zeros((self.square_size, self.square_size))

        # Mark the center location.
        x_center = y_center = int(self.square_size / 2)
        self.center = np.array([x_center, y_center])
        # Fill particle with values.
        # 0  : No occupied.
        # 1  : Holds particle
        matrix = np.zeros((self.square_size, self.square_size))

        matrix[x_center, y_center] = 1  # Center
        self.matrix = matrix
        cmap = colors.ListedColormap(['white', 'black'])
        self.cmap = cmap
        if not os.path.isdir("data"):
            os.mkdir("data")

    def run_dla(self, early_stop=False, radius_threshold=None,save_steps=False):
        """
            run dla executes the simulation model. The parameter have to be defined in the class object.
        @param early_stop:
                Default is false. If True apply early stop.

        @param radius_threshold:
                default threshold is self.radius
                in case you want to stop at any given radius.
        @param save_steps:
                Save output at every 100 particle added as pickle file.
        @return:
            None
        """

        # Initialization

        run_flag = True  # Flag to indicate when the simulation should stop.
        particle_count = 0  # No of particles in cluster.
        max_radius = 0
        
        all_data = [] 
        if radius_threshold == None:
            radius_threshold = self.radius
        while run_flag:
            found_friend = False
            location = self.spawn()

            while not found_friend:
                # Run the checking/walking function
                location_new, found_friend = self.move(location)

                if found_friend and self.decision():

                    point = np.array(location)
                    dist = np.linalg.norm(point - self.center)
                    if (dist >= radius_threshold) and early_stop:
                        run_flag = False
                        print("Early stop Particles attached: ", particle_count + 1)

                    if dist > max_radius:
                        max_radius = dist
                    # current location, replace with 1 and stop
                    self.matrix[location[1]][location[0]] = 1
                    particle_count += 1
                    
                    if (particle_count%100==0) and save_steps:
                        fractal_dimension = self.fractal_dimension(self.matrix)
                        all_data.append([self.radius, max_radius, particle_count, fractal_dimension,self.k])
                # Otherwise, save the location
                else:
                    location = location_new
                
                if particle_count >= self.max_count:
                    print("Particles attached: ", particle_count)
                    run_flag = False

        fractal_dimension = self.fractal_dimension(self.matrix)
        print('r,m_r,N,F,k')
        print(self.radius, max_radius, particle_count, fractal_dimension,self.k)
        all_data.append([self.radius, max_radius, particle_count, fractal_dimension,self.k])
        if save_steps:
            filename = str(particle_count) + '_' + str(self.k) + '_out.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(all_data, f)
        return [self.radius, max_radius, particle_count, fractal_dimension,self.k]

    def spawn(self):
        """
            Spawns the particle are a random location on the edge of the square
        """
        random_side = np.random.randint(4)
        random_point = np.random.randint(self.square_size)
        if random_side == 0:  # left
            location = [0, random_point]
        elif random_side == 1:  # right
            location = [self.square_size - 1, random_point]
        elif random_side == 3:  # top
            location = [random_point, 0]
        else:
            location = [random_point, self.square_size - 1]

        return location

    def move(self, location):
        """
            @param location: current location of the moving particle.
            
            @return:
                    location,foundFriend
                    location : updated location
                    foundFriend : Boolean, True if attached else False
                    
        """

        foundFriend = False  # found another particle

        if location[1] + 1 < self.square_size:
            neighborDown = self.matrix[location[1] + 1, location[0]]
            if neighborDown == 1:
                foundFriend = True

        if location[1] - 1 > 0:
            neighborUp = self.matrix[location[1] - 1, location[0]]
            if neighborUp == 1:
                foundFriend = True

        if location[0] + 1 < self.square_size:
            neighborRight = self.matrix[location[1], location[0] + 1]
            if neighborRight == 1:
                foundFriend = True

        if location[0] - 1 > 0:
            neighborLeft = self.matrix[location[1], location[0] - 1]
            if neighborLeft == 1:
                foundFriend = True

        # After checking locations, if locations are good, start the random walk
        if not foundFriend:
            decide = random.random()

            if decide < 0.25:
                location = [location[0] - 1, location[1]]
            elif decide < 0.5:
                location = [location[0] + 1, location[1]]
            elif decide < 0.75:
                location = [location[0], location[1] + 1]
            else:
                location = [location[0], location[1] - 1]

            location[0] = location[0] % self.square_size
            location[1] = location[1] % self.square_size
            if location[0] < 0:
                location[0] = self.square_size - location[0] - 1
            if location[1] < 0:
                location[1] = self.square_size - location[1] - 1
        return location, foundFriend

    def decision(self):
        """
            @return Boolean : True : with probability self.k else False with 1-self.k
        """
        random.seed()
        return random.random() < self.k

    def fractal_dimension(self, inputMatrix):
        """
            Calculates fraction dimension of a given input matrix.
            # From https://github.com/rougier/np-100 (#87) # Original reference of the code I copied
            @param inputMatrix:
                matrix for which fractal dimension have to be calculated
            @return:
                Calculates fractal dimension
        """
        z = inputMatrix.copy()
        threshold = 1
        # Only for 2d image
        assert (len(z.shape) == 2)

        
        def boxcount(z, k):
            S = np.add.reduceat(
                np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
                np.arange(0, z.shape[1], k), axis=1)

            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k * k))[0])

        # Transform z into a binary array
        z = (z < threshold)

        # Minimal dimension of image
        p = min(z.shape)

        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(p) / np.log(2))

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(z, size))

        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
