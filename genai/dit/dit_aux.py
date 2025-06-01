
import matplotlib.pyplot as plt
import numpy as np

def plot(time_per_batch_list, num_workers_list, y_title="Time per Batch (ms)", image_fn="image"):
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(num_workers_list, time_per_batch_list, 'b-o', linewidth=2, markersize=8)

        # Customize the plot
        plt.title(f'{y_title} vs Number of Workers', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel(y_title, fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add value labels on each point
        for x, y in zip(num_workers_list, time_per_batch_list):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)

        # Improve layout
        plt.tight_layout()

        # Save as image.jpg
        plt.savefig(f'{image_fn}.jpg', dpi=300, bbox_inches='tight', format='jpg')

        # Also save as high-quality PNG (optional)
        plt.savefig(f'{image_fn}.png', dpi=300, bbox_inches='tight', format='png')

        # Display the plot (optional - remove if running in script)
        plt.show()

        print("Plot saved as 'image.jpg' and 'image.png'")
        print(f"X-values (num_workers): {num_workers_list}")
        print(f"Y-values (time_per_batch): {time_per_batch_list}")


def plot2(time_per_batch1, time_per_batch2, num_workers_list, 
          y_title="Time per Batch (ms)", 
          x_title="Number of Workers", 
          image_fn="image",
          batth_sizes=(100,200)):
    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot both lines
    plt.plot(num_workers_list, time_per_batch1, 'b-o', linewidth=2, markersize=8, 
            label=f'(batch_size={batth_sizes[0]}) {y_title}', color='#2E86AB')
    plt.plot(num_workers_list, time_per_batch2, 'r-s', linewidth=2, markersize=8, 
            label=f'(batch_size={batth_sizes[1]}) {y_title}', color='#A23B72')

    # Customize the plot
    plt.title(f'{y_title} Comparison vs {x_title}', fontsize=16, fontweight='bold')
    plt.xlabel(f'{x_title}', fontsize=12)
    plt.ylabel(f'{y_title}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')

    # Add value labels on each point for both lines
    for x, y1, y2 in zip(num_workers_list, time_per_batch1, time_per_batch2):
        plt.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points", 
                    xytext=(0,12), ha='center', fontsize=9, color='#2E86AB')
        plt.annotate(f'{y2:.2f}', (x, y2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='#A23B72')

    # Improve layout
    plt.tight_layout()

    # Save as image.jpg
    plt.savefig(f'{image_fn}.jpg', dpi=300, bbox_inches='tight', format='jpg')

    # Also save as high-quality PNG (optional)
    plt.savefig(f'{image_fn}.png', dpi=300, bbox_inches='tight', format='png')

    # Display the plot (optional - remove if running in script)
    plt.show()

    print("Plot saved as 'image.jpg' and 'image.png'")
    print(f"X-values (num_workers): {num_workers_list}")
    print(f"Y-values (time_per_batch1): {time_per_batch1}")
    print(f"Y-values (time_per_batch2): {time_per_batch2}")