"""
Extract physical quantities from tracker output and inspect them against the
source video.

Publication figure generation lives outside this repository: this tool writes
self-describing HDF5, and plotting scripts read it.
"""

import argparse
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Import the newly created postprocess package
from postprocess import (
    create_quiver_video,
    create_quiver_image,
    create_contour_video,
    create_spacetime_diagram,
    compute_shedding_frequency,
    compute_cavity_metrics,
    extract_line_profiles,
    compute_time_average_fields,
    create_profile_video,
    copy_provenance,
)

def main():
    parser = argparse.ArgumentParser(
        description="Optical Flow Visualization Tool. Generate videos and plots from raw videos & HDF5 results."
    )
    
    # Base inputs (optional for standalone comparison tasks)
    parser.add_argument('--video', help="Path to the original raw video (.avi, .mp4)")
    parser.add_argument('--h5', help="Path to the velocity/uncertainty .h5 file from tracking.py")
    
    # Optional outputs
    parser.add_argument('--out_dir', default='results', help="Directory to save generated visualizations")
    
    # Quiver features
    parser.add_argument('--quiver', action='store_true', help="Generate a quiver (arrow) velocity overlay video")
    parser.add_argument('--quiver_scale', type=float, default=4.0, help="Velocity arrow length multiplier. Default: 4.0")
    parser.add_argument('--quiver_skip', type=int, default=16, help="Pixel spacing between arrows. Default: 16")
    
    # Contour features
    parser.add_argument('--contour', action='store_true', help="Generate a translucent heat-map contour overlay video")
    parser.add_argument('--dataset', type=str, default='uncertainty', choices=['velocity', 'uncertainty'], 
                        help="Which dataset from the HDF5 to render as a contour (velocity or uncertainty)")
    parser.add_argument('--alpha', type=float, default=0.5, help="Heatmap transparency. 0.0=invisible, 1.0=opaque. Default: 0.5")
    
    # Spacetime plot features
    parser.add_argument('--spacetime', action='store_true', help="Generate an average Space-Time (x-t) tracking plot")
    
    # Animated Line Profile features
    parser.add_argument('--profile_video', action='store_true', help="Generate an animated video of 10 equidistant instantaneous U-profiles overlaid on the velocity contour")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for generated videos (profile_video, contour, quiver). Default: 10")
    
    # Analytics features
    parser.add_argument('--average', action='store_true', help="Compute and output full 2D time-averaged velocity and Reynolds Stresses")
    parser.add_argument('--frequency', action='store_true', help="Compute and output the cavity shedding frequency (PSD)")
    parser.add_argument('--cavity', action='store_true', help="Compute and output the time-averaged cavity variance and length limits (in mm)")
    
    # Quantitative Data Extraction
    parser.add_argument('--profiles', type=float, nargs='+', help="List of x-locations (in mm) from throat to extract mean velocity and Reynolds stresses.")
    parser.add_argument('--angle', type=float, default=0.0,
                        help="Deprecated. Align the field at tracking time with "
                             "tracker.py --rotate_angle; a non-zero value here is an error.")
    parser.add_argument('--frame', type=int, default=None, help="Extract profiles at a specific instantaneous frame instead of time-averaged.")
    
    parser.add_argument('--plot', action='store_true', help="Generate and save PNG plots for analytical flags like --frequency, --cavity, or --profiles")

    args = parser.parse_args()
    
    needs_video = args.quiver or args.contour or args.spacetime or args.profile_video \
        or args.frequency or args.cavity
    if not args.h5:
        parser.error("--h5 is required.")
    if needs_video and not args.video:
        parser.error("--video is required for the requested operation.")
    
    # Prepare the output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")
        
    # Get a base name for the output files
    if args.h5:
        base_name = os.path.splitext(os.path.basename(args.h5))[0]
    elif args.video:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
    else:
        base_name = "output"
    
    print("\n--- Starting Post-Processing Visualization ---")

    if args.quiver:
        if args.frame is not None:
            out_path = os.path.join(args.out_dir, f"{base_name}_frame_{args.frame}_quiver.png")
            print(f"\n[1] Generating Instantaneous Quiver Image (Frame {args.frame}) --> {out_path}")
            create_quiver_image(args.video, args.h5, out_path, frame_idx=args.frame, scale=args.quiver_scale, skip=args.quiver_skip)
        else:
            out_path = os.path.join(args.out_dir, f"{base_name}_quiver.mp4")
            print(f"\n[1] Generating Quiver Video --> {out_path}")
            create_quiver_video(args.video, args.h5, out_path, scale=args.quiver_scale, skip=args.quiver_skip, fps=args.fps)
        
    if args.contour:
        out_path = os.path.join(args.out_dir, f"{base_name}_{args.dataset}_contour.mp4")
        print(f"\n[2] Generating {args.dataset.capitalize()} Contour Video --> {out_path}")
        create_contour_video(args.video, args.h5, dataset_name=args.dataset, output_path=out_path, alpha=args.alpha, fps=args.fps)
        
    if args.profile_video:
        out_path = os.path.join(args.out_dir, f"{base_name}_profile_anim.mp4")
        print(f"\n[VP] Generating Profile Overlay Video --> {out_path}")
        create_profile_video(args.video, args.h5, output_path=out_path, num_profiles=10, fps=args.fps)
        
    if args.spacetime:
        # Load ROI automatically from HDF5 metadata
        with h5py.File(args.h5, 'r') as f:
            if 'roi' in f['velocity'].attrs:
                roi = tuple(f['velocity'].attrs['roi'])
                print(f"Loaded ROI from HDF5: {roi}")
            else:
                roi = [0, -1, 0, -1]
                print(f"ROI not found in HDF5, using full frame: {roi}")

        out_path = os.path.join(args.out_dir, f"{base_name}_spacetime.png")
        print(f"\n[3] Generating Space-Time (x-t) Diagram --> {out_path}")
        xt_image = create_spacetime_diagram(args.video, roi)
        
        # Save diagram using matplotlib
        plt.figure(figsize=(12, 6), layout='constrained')
        
        plt.imshow(xt_image.T, aspect='auto', cmap='gray')
        plt.title('Cavity Shedding Space-Time (x-t) Diagram')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Space (Pixels)')
        plt.colorbar(label='Averaged Brightness')
        plt.savefig(out_path, dpi=300, transparent=True)
        plt.close()
        
    if args.frequency:
        # Load ROI automatically from HDF5 metadata
        with h5py.File(args.h5, 'r') as f:
            attrs = f['velocity'].attrs
            roi = tuple(attrs['roi']) if 'roi' in attrs else None
            fps_capture = float(attrs['fps_capture']) if 'fps_capture' in attrs else None

        if fps_capture is None:
            print("Error: the HDF5 file has no 'fps_capture' attribute. "
                  "Re-run tracker.py, or pass the capture rate explicitly.")
            sys.exit(1)
        
        signal, f_arr, Pxx, dom_freq = compute_shedding_frequency(args.video, roi=roi, fps=fps_capture)
        print(f"\n[4] Dominant Shedding Frequency: {dom_freq:.2f} Hz")
        
        if args.plot:
            out_path = os.path.join(args.out_dir, f"{base_name}_frequency.png")
            print(f"    --> Saving Frequency Plot to {out_path}")
            
            plt.figure(figsize=(10, 5), layout='constrained')
            plt.plot(f_arr, Pxx)
            
            # Marking the peak value
            peak_idx = np.argmax(Pxx)
            plt.scatter(f_arr[peak_idx], Pxx[peak_idx], color='red', zorder=5)
            plt.annotate(f"{dom_freq:.2f} Hz", (f_arr[peak_idx], Pxx[peak_idx]), 
                         textcoords="offset points", xytext=(10,10), ha='left', color='red', weight='bold')
                         
            plt.title(f'Power Spectral Density (Dominant Frequency: {dom_freq:.1f} Hz)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.xlim(0, max(f_arr)/10) # Zoom into lower frequencies usually
            plt.grid(True)
            plt.savefig(out_path, dpi=300, transparent=True)
            plt.close()
            
    if args.cavity:
        # Get calibration factor from HDF5
        mm_per_px = 1.0
        with h5py.File(args.h5, 'r') as f:
            if 'mm_per_px' in f['velocity'].attrs:
                mm_per_px = f['velocity'].attrs['mm_per_px']
                print(f"\n[5] Calculating cavity length (Calibration: px = {mm_per_px:.6f} mm)")
            else:
                print(f"\n[5] Calculating cavity length (No calibration found, using pixels)")
                
        avg, var, cavity_mask, max_indx = compute_cavity_metrics(args.video)
        
        if max_indx is not None:
            length_px = max_indx[1]
            length_mm = length_px * mm_per_px
            print(f"    Cavity Length: {length_mm:.2f} mm ({length_px} px)")
        else:
            print("    Could not determine cavity length (variance threshold not met).")
        
        if args.plot and max_indx is not None:
            out_path = os.path.join(args.out_dir, f"{base_name}_cavity.png")
            print(f"    --> Saving Cavity Plot to {out_path}")
            
            plt.figure(figsize=(10, 4), layout='constrained')
            
            # Show mean plot
            plt.imshow(avg, cmap='gray')
            
            # Contour representing the cavity length
            if cavity_mask is not None:
                # Calculate coordinates for drawing the contour line
                plt.contour(cavity_mask, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.8)
                
            # Location of max std shown on mean plot
            plt.scatter(max_indx[1], max_indx[0], color='red', marker='x', s=100, label='Location of Max Standard Deviation')
            plt.legend(loc='upper right')
            
            plt.title(f'Time-Averaged Brightness with Cavity Extent ({length_mm:.2f} mm)')
            plt.axis('off')
            plt.savefig(out_path, dpi=300, transparent=True)
            plt.close()
            
    if args.average:
        print(f"\n[7] Computing Full 2D Time-Averaged Fields")
        avg_fields = compute_time_average_fields(args.h5)
        
        out_path_h5 = os.path.join(args.out_dir, f"{base_name}_average.h5")
        
        with h5py.File(out_path_h5, 'w') as fout:
            with h5py.File(args.h5, 'r') as fin:
                copy_provenance(dict(fin['velocity'].attrs), fout)
            for key, data in avg_fields.items():
                fout.create_dataset(key, data=data)

        print(f"    --> Saved 2D averages to: {out_path_h5}")
        
    if args.profiles:
        if args.frame is not None:
            print(f"\n[6] Extracting instantaneous velocity profiles at X = {args.profiles} mm (frame {args.frame})")
        else:
            print(f"\n[6] Extracting time-averaged velocity profiles at X = {args.profiles} mm")
            
        results, lines_h5_path = extract_line_profiles(
            args.h5, args.profiles, angle_deg=args.angle, frame_idx=args.frame)

        print(f"    --> Saved line extractions to: {lines_h5_path}")
        for x_wall, data in results.items():
            n = data['n_samples']
            print(f"        X = {x_wall} mm | {int(np.nanmedian(n))} valid samples (median over the profile)")
        
    if not (args.quiver or args.contour or args.spacetime or args.frequency
            or args.cavity or args.profiles or args.average or args.profile_video):
        print("\nNo operation requested.")
        print("Extraction: --profiles, --average, --frequency, --cavity")
        print("Inspection: --quiver, --contour, --profile_video, --spacetime")
        print("Example: python3 visualize_results.py --video run.cine --h5 run_raft.h5 --average")
    else:
        print("\n--- Processing Complete ---")

if __name__ == '__main__':
    main()
