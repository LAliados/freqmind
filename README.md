# freqmind
Advanced Linux governor based on PMU events and a little bit ML

# Requirements
1. Linux kernel > 6.0
2. AMD processor with Zen 2 architecture
# Building & Running
1. Install python with tensorflow
2. Install build-essentials and Linux kernel headers
3. Run `make train`
4. Run `sudo insmod build/freqmind.ko`
5. Run `echo freqmind_train | sudo tee /sys/devices/system/cpu/cpufreq/policy2/scaling_governor`
6. To check results run `dmesg -w`