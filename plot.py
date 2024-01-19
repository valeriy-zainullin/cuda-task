data = list(filter(lambda x: x, """
Subplot 1 (time as a variable of number of threads).
X label: Number of threads
Y label: Time
Curve 1: GPU time
Curve 2: CPU time (single thread)
# x = 360, y1 = 641.415ms, y2 = 873.467ms
# x = 750, y1 = 591.711ms, y2 = 897.755ms
# x = 1500, y1 = 570.667ms, y2 = 800.563ms
# x = 600, y1 = 602.818ms, y2 = 877.14ms
# x = 1250, y1 = 573.586ms, y2 = 800.883ms
# x = 2500, y1 = 560.034ms, y2 = 800.347ms
# x = 1200, y1 = 573.749ms, y2 = 805.559ms
# x = 2500, y1 = 560.066ms, y2 = 800.207ms
# x = 5000, y1 = 552.566ms, y2 = 804.551ms

Subplot 2 (time as a variable of number of items).
X label: Number of items
Y label: Time
Curve 1: GPU time
Curve 2: CPU time (single thread)
# x = 1, y1 = 0.040096ms, y2 = 0.002144ms
# x = 100, y1 = 0.023232ms, y2 = 0.002368ms
# x = 1000, y1 = 0.025312ms, y2 = 0.009312ms
# x = 100000, y1 = 0.578336ms, y2 = 0.766944ms
# x = 1000000, y1 = 5.60634ms, y2 = 7.9616ms
# x = 100000000, y1 = 555.403ms, y2 = 803.363ms
""".split('\n')))

print(data)
exit()

import matplotlib.pyplot as plt
import sys

num_subplots = 2

#data = open(sys.argv[

x_values = []
y_values = []

for plot_index, num_segments in enumerate([1000, 10**6, 10**8]):
	x_values = []
	y_values = []
	for num_processes in range(1, 8+1):
		print(num_segments, num_processes)
		script_name = "./cluster_run2.sbatch.p=%d.sh" % num_processes

		with open(script_name, 'w') as stream:
			stream.write(sbatch_script_pattern.format(num_processes))
		cmd = ["sbatch", "-v", "--export=NUM_SEGMENTS=%d" % num_segments, script_name]
		subprocess.check_output(cmd)

		# Wait for the job to finish
		while True:
			num_jobs_active = subprocess.check_output("squeue -n vzainullinMPI | wc -l", shell=True).decode("utf-8")
			num_jobs_active = int(num_jobs_active)
			if num_jobs_active == 1:
				break
			time.sleep(0.5)

		output = None
		with open("out.txt", 'r') as stream:
			output = stream.read()

		result = None
		for line in output.split('\n'):
			if "acceleration" in line:
				_, value = line.split('=')
				result = float(value)
		assert result is not None

		x_values.append(num_processes)
		y_values.append(result)
		print("acceleration =", result)

		print("▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")

	plt.plot(x_values, y_values, label="N = %1.e" % num_segments)

plt.legend(fontsize = 12)
plt.savefig("Acceleration.jpg")
