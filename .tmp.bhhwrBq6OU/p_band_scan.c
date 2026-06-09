#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <pthread.h>

#include "filter.h"
#include "signal.h"
#include "timing.h"

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0

void usage() {
  printf("usage: band_scan text|bin|mmap signal_file Fs filter_order num_bands\n");
}

double avg_power(double* data, int num) {

  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }

  return ss / num;
}

double max_of(double* data, int num) {

  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {

  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {

  double dc = avg_of(data,num);

  printf("Removing DC component of %lf\n",dc);

  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}

// struct
typedef struct {
  signal *sig;
  int filter_order;
  double bandwidth;
  int band_start;
  int band_end;
  double *band_power;
  int processor_id;
} thread_args;

// Function run by each thread (boiler plate from band_scan.c) 
void* worker(void* arg) {
  thread_args* a = (thread_args*)arg; 

  // put ourselves on the desired processor
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(a->processor_id, &set); // sched_setaffinity -> pthread_setaffinity_np
  if (sched_setaffinity(0,sizeof(set),&set) < 0) { // do it
    perror("Can't setaffinity");  // hopefully doesn't fail
    exit(-1);
  }

  double filter_coeffs[a->filter_order + 1];

  for (int band = a->band_start; band < a->band_end; band++) {
    // making the filter
    generate_band_pass(a->sig->Fs, band * a->bandwidth + 0.0001, (band + 1) * a->bandwidth - 0.0001, a->filter_order, filter_coeffs);
    generate_band_pass(a->sig->Fs, band * a->bandwidth + 0.0001, (band + 1) * a->bandwidth - 0.0001, a->filter_order, filter_coeffs);
    hamming_window(a->filter_order, filter_coeffs);
    // convolve
    convolve_and_compute_power(a->sig->num_samples, a->sig->data, a->filter_order, filter_coeffs, &(a->band_power[band]));
  }

  pthread_exit(NULL); // finish - no return value
}

int analyze_signal(signal* sig, int filter_order, int num_bands, int num_threads, int num_processors, double* lb, double* ub) {

  double Fc        = (sig->Fs) / 2;
  double bandwidth = Fc / num_bands;

  remove_dc(sig->data,sig->num_samples);

  double signal_power = avg_power(sig->data,sig->num_samples);

  printf("signal average power:     %lf\n", signal_power);

  resources rstart;
  get_resources(&rstart,THIS_PROCESS);
  double start = get_seconds();
  unsigned long long tstart = get_cycle_count();

  double band_power[num_bands];
  pthread_t threads[num_threads];
  thread_args args[num_threads];
  
  int bands_per_thread = num_bands / num_threads;
  int leftover = num_bands % num_threads;
  int band_start = 0;

  for (int t = 0; t < num_threads; t++) {
    int my_bands = bands_per_thread + (t < leftover ? 1 : 0);
    args[t].sig          = sig;
    args[t].filter_order = filter_order;
    args[t].bandwidth    = bandwidth;
    args[t].band_start   = band_start;
    args[t].band_end     = band_start + my_bands;
    args[t].band_power   = band_power;
    args[t].processor_id = t % num_processors;
    pthread_create(&threads[t], NULL, worker, &args[t]);
    band_start += my_bands;
  }
  // waiting for all threads to finish
  for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);

  unsigned long long tend = get_cycle_count();
  double end = get_seconds();

  resources rend;
  get_resources(&rend,THIS_PROCESS);

  resources rdiff;
  get_resources_diff(&rstart, &rend, &rdiff);

  // Pretty print results
  double max_band_power = max_of(band_power,num_bands);
  double avg_band_power = avg_of(band_power,num_bands);
  int wow = 0;
  *lb = -1;
  *ub = -1;

  for (int band = 0; band < num_bands; band++) {
    double band_low  = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;

    printf("%5d %20lf to %20lf Hz: %20lf ",
           band, band_low, band_high, band_power[band]);

    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
      printf("*");
    }

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

      // band of interest
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (*lb < 0) {
          *lb = band * bandwidth + 0.0001;
        }
        *ub = (band + 1) * bandwidth - 0.0001;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }

    printf("\n");
  }

  printf("Resource usages:\n\
User time        %lf seconds\n\
System time      %lf seconds\n\
Page faults      %ld\n\
Page swaps       %ld\n\
Blocks of I/O    %ld\n\
Signals caught   %ld\n\
Context switches %ld\n",
         rdiff.usertime,
         rdiff.systime,
         rdiff.pagefaults,
         rdiff.pageswaps,
         rdiff.ioblocks,
         rdiff.sigs,
         rdiff.contextswitches);

  printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
         "Note that cycle count only makes sense if the thread stayed on one core\n",
         tend - tstart, cycles_to_seconds(tend - tstart), timing_overhead());
  printf("Analysis took %lf seconds by basic timing\n", end - start);

  return wow;
}

int main(int argc, char* argv[]) {

  if (argc != 8) {
    usage();
    return -1;
  }

  char sig_type    = toupper(argv[1][0]);
  char* sig_file   = argv[2];
  double Fs        = atof(argv[3]);
  int filter_order = atoi(argv[4]);
  int num_bands    = atoi(argv[5]);
  // add the 2 new args
  int num_threads    = atoi(argv[6]);
  int num_processors = atoi(argv[7]);

  assert(Fs > 0.0);
  assert(filter_order > 0 && !(filter_order & 0x1));
  assert(num_bands > 0);
  // addeded 2 new asserts
  assert(num_threads > 0);
  assert(num_processors > 0);

  printf("type:     %s\n\
file:     %s\n\
Fs:       %lf Hz\n\
order:    %d\n\
bands:    %d\n",
         sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
         sig_file,
         Fs,
         filter_order,
         num_bands);

  printf("Load or map file\n");

  signal* sig;
  switch (sig_type) {
    case 'T':
      sig = load_text_format_signal(sig_file);
      break;

    case 'B':
      sig = load_binary_format_signal(sig_file);
      break;

    case 'M':
      sig = map_binary_format_signal(sig_file);
      break;

    default:
      printf("Unknown signal type\n");
      return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;

  double start = 0;
  double end   = 0;
  if (analyze_signal(sig, filter_order, num_bands, num_threads, num_processors, &start, &end)) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", start, end, (end + start) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);

  return 0;
}

