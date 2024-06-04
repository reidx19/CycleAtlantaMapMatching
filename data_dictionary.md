Notebook Output Descriptions

Raw Files (SQL Dump)
- coordsX.csv: csvs of all the trajectories
- user.csv: csv of all the users
- trip.csv: csv of all the trips

After process_coords_and_trips.ipynb
- coords_0.pkl: dictionary containing all the trajectories keyed by tripid
- trips_0.pkl: dataframe containing trips and relevent information

After process_users.ipynb
- users_0.pkl: dataframe containing user info, users that are not in trips_0.pkl are removed
- users_1.pkl: merges users with the same email address (userid is mapped to the first appearing userid)
- trips_1.pkl: adds new column called userid_remap that is the result of the merge done in users_1.pkl AND removes trips when the userid is not present in users_0.pkl

After gps_simplifications.ipynb
- rdp.pkl: dictionary of trajectories that have been simiplified using the Ramer–Douglas–Peucker algorithm
- reduced_spacing.pkl: dictionary of trajectories that have had their point resolution reduced using the spacing between points

After redundant_trips.ipynb
